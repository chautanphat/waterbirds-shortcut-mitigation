/* Shared Mixer primitives. Existing canvas behavior is intentionally preserved. */
(function(global){
function aligned(im,w,h,fit){const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d',{willReadFrequently:true});if(fit==='stretch')x.drawImage(im,0,0,w,h);else{const scale=(fit==='contain'?Math.min:Math.max)(w/im.naturalWidth,h/im.naturalHeight);const dw=im.naturalWidth*scale,dh=im.naturalHeight*scale;x.drawImage(im,(w-dw)/2,(h-dh)/2,dw,dh)}return x.getImageData(0,0,w,h)}
function blend(a,b,t){const out=new Uint8ClampedArray(a.length);for(let i=0;i<a.length;i+=4){const aa=a[i+3]/255*t,ba=b[i+3]/255*(1-t),alpha=aa+ba;for(let k=0;k<3;k++)out[i+k]=alpha?(a[i+k]*aa+b[i+k]*ba)/alpha:0;out[i+3]=alpha*255}return out}
global.MixerCore={aligned,blend};
})(globalThis);
