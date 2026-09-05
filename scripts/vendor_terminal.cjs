const fs = require('fs');
const path = require('path');
const out = path.join(__dirname, '../dashboard/static/vendor');
fs.mkdirSync(out, {recursive:true});
for (const [src, dest] of [
  ['@xterm/xterm/lib/xterm.js','xterm.js'],
  ['@xterm/xterm/css/xterm.css','xterm.css'],
  ['@xterm/xterm/LICENSE','xterm-LICENSE'],
  ['@xterm/addon-fit/lib/addon-fit.js','addon-fit.js'],
  ['@xterm/addon-fit/LICENSE','addon-fit-LICENSE'],
]) fs.copyFileSync(path.join(__dirname,'../node_modules',src),path.join(out,dest));
