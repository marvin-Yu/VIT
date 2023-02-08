# GreenNet project
```shell
wget http://crt-e302.sh.intel.com/files/greennet/intel.zip
unzip -d models intel.zip

#run benchmark
python benchmarks.py
```

# Current issue
- product: `python benchmarks.py --models=product`
RuntimeError: index out of range in self
```shell
        File "code/__torch__/torch/nn/modules/sparse/___torch_mangle_1.py", line 9, in forward
        input: Tensor) -> Tensor:
        weight = self.weight
        return torch.embedding(weight, input)
            ~~~~~~~~~~~~~~~ <--- HERE
        ......
```
- yuanjun: `python benchmarks.py --models=yuanjun`
RuntimeError: PyTorch is not linked with support for cuda devices
```shell
        Traceback of TorchScript, serialized code (most recent call last):
    File "code/__torch__/vitstr_ada_wh_res18_hv_clip_nopos_light.py", line 155, in forward
        _10 = torch.transpose(_9, 1, 2)
        dir_tokens0 = torch.to(dir_tokens, 6)
        dir_tokens1 = torch.to(dir_tokens0, dtype=6, layout=0, device=torch.device("cuda"), pin_memory=False)
                    ~~~~~~~~ <--- HERE
        x00 = torch.cat([dir_tokens1, _10], 1)
        bias13 = ln_pre.bias
        ......
```
- zk: `python benchmarks.py --models=zk`
NotImplementedError: The following operation failed in the TorchScript interpreter.
```shell
        NotImplementedError: The following operation failed in the TorchScript interpreter.
        Traceback of TorchScript, serialized code (most recent call last):
        File "code/__torch__.py", line 25, in forward
            batch_enumeration = torch.unsqueeze(_9, 1)
            _10 = torch.slice(_4, 1, 0, 9223372036854775807)
            _11 = torch.to(batch_enumeration, dtype=4, layout=0, device=torch.device("cuda:0"))
                ~~~~~~~~ <--- HERE
            x_of_kp0 = torch.to(x_of_kp, dtype=4, layout=0, device=torch.device("cuda:0"))
            y_of_kp0 = torch.to(y_of_kp, dtype=4, layout=0, device=torch.device("cuda:0"))

        Traceback of TorchScript, original code (most recent call last):
        export_lg_model.py(188): forward
        /home/sunjinfeng.sjf/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(860): _slow_forward
        /home/sunjinfeng.sjf/.local/lib/python3.6/site-packages/torch/nn/modules/module.py(887): _call_impl
        /home/sunjinfeng.sjf/.local/lib/python3.6/site-packages/torch/jit/_trace.py(940): trace_module
        /home/sunjinfeng.sjf/.local/lib/python3.6/site-packages/torch/jit/_trace.py(742): trace
        /usr/local/lib/python3.6/site-packages/quake_tools/export/torch_utils.py(140): gen_model_files
        /usr/local/lib/python3.6/site-packages/quake_tools/export/torch_utils.py(278): normal_export
        export_lg_model.py(232): <module>
        RuntimeError: Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::empty_strided' is only available for these backends: [CPU, Meta, QuantizedCPU, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMeta, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradNestedTensor, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PythonDispatcher].
        ......
```