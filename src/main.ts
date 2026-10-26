async function init() {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  if (!navigator.gpu) {
    alert('WebGPU is not supported in this browser. Please use Chrome/Edge with WebGPU enabled.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error('Failed to get WebGPU adapter.');
    return;
  }
  const device = await adapter.requestDevice();

  device.lost.then((info) => {
    console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
  });

  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  // Keep the drawing buffer matched to the canvas's on-screen size
  const resizeObserver = new ResizeObserver(() => {
    const dpr = window.devicePixelRatio || 1;
    const maxDim = device.limits.maxTextureDimension2D;
    canvas.width = Math.max(1, Math.min(Math.floor(canvas.clientWidth * dpr), maxDim));
    canvas.height = Math.max(1, Math.min(Math.floor(canvas.clientHeight * dpr), maxDim));
  });
  resizeObserver.observe(canvas);

  console.log('WebGPU successfully initialized! Device:', device);

  function frame() {
    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.02, g: 0.03, b: 0.06, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

init();
