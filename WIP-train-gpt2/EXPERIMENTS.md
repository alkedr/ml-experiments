NanoGPT unmodified:
- error about 1060 being too old for Triton
- batch_size=1 for VRAM
- 850ms per iteration, mfu 0.4%

NanoGPT fp32:
- 620ms per iteration, mfu 0.6% (1.8 TFLOPS out of 4.375 for fp32 on 1060)
