// PM2 config for the allocator bridge
// Install: cd /path/to/regime-aware-strategy-allocator && pm2 start live/ecosystem.config.js
module.exports = {
  apps: [{
    name: "allocator-bridge",
    script: "live/bridge.py",
    interpreter: ".venv/bin/python3",
    cwd: process.env.ALLOCATOR_DIR || process.cwd(),
    args: "--loop 300",
    autorestart: true,
    max_restarts: 10,
    min_uptime: "30s",
    env: {
      PYTHONPATH: process.env.ALLOCATOR_DIR || process.cwd(),
      SINGULARITY_DIR: process.env.SINGULARITY_DIR || "",
      DIVERGENCE_DIR:  process.env.DIVERGENCE_DIR  || "",
      HORIZON_DIR:     process.env.HORIZON_DIR     || "",
      ALLOCATOR_STATE: process.env.ALLOCATOR_STATE || "./state/allocator_output.json",
    }
  }]
};
