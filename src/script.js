const canvas = document.getElementById('aurora-canvas');
const ctx = canvas.getContext('2d');

let width, height;

function resize() {
  width = canvas.width = window.innerWidth;
  height = canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

// --- 1. FULL-SCREEN CELESTIAL SKYBOX (650 Stars) ---
const starCount = 650;
const stars = Array.from({ length: starCount }, () => {
  const isTwinkling = Math.random() > 0.86; 
  return {
    x: Math.random() * width,
    y: Math.random() * height, // FIXED: Spread entirely from top to bottom
    size: Math.random() * 1.3 + 0.2,
    alpha: Math.random() * 0.7 + 0.3, 
    isTwinkling: isTwinkling,
    speed: isTwinkling ? (0.0005 + Math.random() * 0.0012) : 0 
  };
});

let time = 0;

function animate() {
  // --- 2. THE SEAMLESS TWILIGHT VOID ---
  const skyGrad = ctx.createLinearGradient(0, 0, 0, height);
  
  skyGrad.addColorStop(0, '#090210');    
  skyGrad.addColorStop(0.12, '#150524'); 
  skyGrad.addColorStop(0.50, '#2b0b36'); 
  skyGrad.addColorStop(0.88, '#421239'); 
  skyGrad.addColorStop(1, '#5c204d');    

  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1.0;
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, width, height);

  // Render optimized star matrix
  ctx.fillStyle = 'white';
  stars.forEach(s => {
    if (s.isTwinkling) {
      s.alpha += s.speed;
      if (s.alpha > 1 || s.alpha < 0.15) s.speed = -s.speed; 
    }
    ctx.globalAlpha = Math.abs(s.alpha);
    ctx.fillRect(s.x, s.y, s.size, s.size); 
  });

  // --- 3. FLAWLESS VOLUMETRIC AURORA CURTAINS ---
  ctx.globalCompositeOperation = 'screen';
  ctx.filter = 'blur(60px)';

  const curtains = 3;
  const centerY = height * 0.43; 

  for (let i = 0; i < curtains; i++) {
    const speed = time * (0.12 + i * 0.08);
    const thickness = 135 + i * 40; 

    ctx.beginPath();
    
    // FIXED: Stepping by 40px keeps it performant but smooths out the low-poly jagged edges
    for (let x = -150; x <= width + 150; x += 40) { 
      const y = centerY 
              + Math.sin(x * 0.0018 + speed + i * 1.5) * 80 
              + Math.cos(x * 0.001 - speed) * 35 
              - thickness;
      
      if (x === -150) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    // Bottom edge curve mapping
    for (let x = width + 150; x >= -150; x -= 40) {
      const y = centerY 
              + Math.sin(x * 0.0018 + speed + i * 1.5) * 80 
              + Math.cos(x * 0.001 - speed) * 35 
              + thickness;
      ctx.lineTo(x, y);
    }
    ctx.closePath();

    const hue = 305 + Math.sin(time * 1.4 + i) * 35; 
    
    // FIXED: Expanded the gradient bounding box (+ 140) so the wave never gets its head chopped off
    const auroraGrad = ctx.createLinearGradient(0, centerY - thickness - 140, 0, centerY + thickness + 140);
    auroraGrad.addColorStop(0, `hsla(${hue}, 100%, 65%, 0)`);
    auroraGrad.addColorStop(0.5, `hsla(${hue}, 100%, 72%, 0.34)`); 
    auroraGrad.addColorStop(1, `hsla(${hue}, 100%, 65%, 0)`);

    ctx.fillStyle = auroraGrad;
    ctx.fill();
  }

  ctx.filter = 'none';
  time += 0.0025; 
  requestAnimationFrame(animate);
}

animate();