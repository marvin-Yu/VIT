#include <cassert>
#include <vector>
#include <string>
#include <unordered_map>

#include <torch/extension.h>

#include "bert_layer_fp32.h"
#include "bert_layer_int8.h"
#include "bert_quantize.h"
#include "bert_util.h"

class BertEncoder;
BertContext* ctx;
BertEncoder* bert;

class BertEncoder {
public:
  BertEncoder(BertContext *ctx, int num_hidden_layers, int max_token_size = 64)
    : _num_hidden_layers(num_hidden_layers),
      _max_token_size(max_token_size) {
    this->is_int8 = ctx->is_int8;
    if (is_int8) {
      int8_bert_layers.resize(_num_hidden_layers);
      for (int i = 0; i < _num_hidden_layers; ++i) {
        int8_bert_layers[i] = new Int8BertLayer(ctx, i);
      }
    } else {
      fp32_bert_layers.resize(_num_hidden_layers);
      for (int i = 0; i < _num_hidden_layers; ++i) {
        fp32_bert_layers[i] = new BatchBertLayer(ctx, i);
      }
    }
  }

  ~BertEncoder() {
    for (int i = 0; i < fp32_bert_layers.size(); ++i) {
      delete fp32_bert_layers[i];
    }
    for (int i = 0; i < int8_bert_layers.size(); ++i) {
      delete int8_bert_layers[i];
    }
  }

  torch::Tensor forward(torch::Tensor &input) {
    assert(input.dim() == 3);
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);

    // Enlarge the buffer if needed
    ctx->resize(batch_size, seq_len);

    float* input_data = input.data_ptr<float>();
    hpj::Matrix<float> input_matrix(input_data, batch_size * seq_len, hidden_size, hidden_size);
    hpj::Matrix<float> *m_data = &input_matrix;

    torch::Tensor output = torch::empty({batch_size, seq_len, hidden_size});
    float* output_data = output.data_ptr<float>();
    hpj::Matrix<float> out_matrix(output_data, batch_size * seq_len, hidden_size, hidden_size);

    if (is_int8) {
      // Quantize input
      uint8_t *y = reinterpret_cast<uint8_t *>(ctx->embQuantBuffer.Data());
      ctx->embQuantBuffer.SetQScheme(hpj::per_tensor_affine);
      float *scales = ctx->embQuantBuffer.Scales();
      int32_t *zp = ctx->embQuantBuffer.ZeroPoint();
      int stride = ctx->embQuantBuffer.Stride();
      QuantizeUtil::quantize_input(input_data, y, scales, zp,
                                   batch_size * seq_len, hidden_size,
                                   hidden_size, stride);

      for (int i = 0; i < _num_hidden_layers - 1; ++i) {
        hpj::Matrix<float> &outBuffer = (i % 2 == 0 ? ctx->outBuffer1 : ctx->outBuffer2);
        int8_bert_layers[i]->forward(*m_data, outBuffer, ctx->embQuantBuffer, ctx->embQuantBuffer);
        m_data = &outBuffer;
      }

      // Last layer, copy result to output tensor
      int last = _num_hidden_layers - 1;
      int8_bert_layers[last]->forward(*m_data, out_matrix, ctx->embQuantBuffer, ctx->embQuantBuffer);
    }
    // FP32
    else {
      for (int i = 0; i < _num_hidden_layers - 1; ++i) {
        hpj::Matrix<float> &outBuffer = (i % 2 == 0 ? ctx->outBuffer1 : ctx->outBuffer2);
        fp32_bert_layers[i]->forward(*m_data, outBuffer);
        m_data = &outBuffer;
      }
      
      // Last layer, copy result to output tensor
      int last = _num_hidden_layers - 1;
      fp32_bert_layers[last]->forward(*m_data, out_matrix);
    }

    return output;
  }

  void init_weights(std::unordered_map<std::string, torch::Tensor> &weights, const std::string &encoder_prefix) {
    for (int i = 0; i < _num_hidden_layers; ++i) {
      std::string prefix = encoder_prefix + ".layer." + std::to_string(i);
      float *queryW = weights[prefix + ".attention.attention.query.weight"].data_ptr<float>();
      float *queryB = weights[prefix + ".attention.attention.query.bias"].data_ptr<float>();
      float *keyW = weights[prefix + ".attention.attention.key.weight"].data_ptr<float>();
      float *keyB = weights[prefix + ".attention.attention.key.bias"].data_ptr<float>();
      float *valueW = weights[prefix + ".attention.attention.value.weight"].data_ptr<float>();
      float *valueB = weights[prefix + ".attention.attention.value.bias"].data_ptr<float>();

      float *att_dense_w = weights[prefix + ".attention.output.dense.weight"].data_ptr<float>();
      float *att_dense_b = weights[prefix + ".attention.output.dense.bias"].data_ptr<float>();

      float *gamma1 = weights[prefix + ".layernorm_before.weight"].data_ptr<float>();
      float *beta1 = weights[prefix + ".layernorm_before.bias"].data_ptr<float>();

      float *intermediateW = weights[prefix + ".intermediate.dense.weight"].data_ptr<float>();
      float *intermediateB = weights[prefix + ".intermediate.dense.bias"].data_ptr<float>();

      float *outputW = weights[prefix + ".output.dense.weight"].data_ptr<float>();
      float *outputB = weights[prefix + ".output.dense.bias"].data_ptr<float>();

      float *gamma2 = weights[prefix + ".layernorm_after.weight"].data_ptr<float>();
      float *beta2 = weights[prefix + ".layernorm_after.bias"].data_ptr<float>();

      if (is_int8) {
        int8_bert_layers[i]->setWeights(queryW, queryB,
                                        keyW, keyB,
                                        valueW, valueB,
                                        att_dense_w, att_dense_b,
                                        gamma1, beta1,
                                        intermediateW, intermediateB,
                                        outputW, outputB,
                                        gamma2, beta2);
      } else {
        fp32_bert_layers[i]->setWeights(queryW, queryB,
                                        keyW, keyB,
                                        valueW, valueB,
                                        att_dense_w, att_dense_b,
                                        gamma1, beta1,
                                        intermediateW, intermediateB,
                                        outputW, outputB,
                                        gamma2, beta2);
      }
    }
  }

private:
  int _hidden_size;
  int _intermediate_size;
  int _num_attention_heads;
  int _num_hidden_layers;
  int _max_token_size;

  bool is_int8;
  std::vector<BatchBertLayer*> fp32_bert_layers;
  std::vector<Int8BertLayer*>  int8_bert_layers;
};

void bert_init(std::unordered_map<std::string, int> configs) {
  int hidden_size = configs["hidden_size"];
  int intermediate_size = configs["intermediate_size"];
  int num_attention_heads = configs["num_attention_heads"];
  int num_hidden_layers = configs["num_hidden_layers"];
  int int8_flag = configs["is_int8"];
  bool is_int8 = (int8_flag != 0);
  ctx = new BertContext(hidden_size, num_attention_heads, intermediate_size, is_int8);
  bert = new BertEncoder(ctx, num_hidden_layers);
}

torch::Tensor bert_forward(torch::Tensor input) {
  return bert->forward(input);
}

void bert_init_weights(std::unordered_map<std::string, torch::Tensor> weights, std::string encoder_prefix) {
  bert->init_weights(weights, encoder_prefix);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &bert_init, "BERT init");
  m.def("init_weights", &bert_init_weights, "BERT init weights");
  m.def("forward", &bert_forward, "BERT forward");
}
