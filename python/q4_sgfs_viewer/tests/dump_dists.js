// Prints the distances to the centre of the alive seats after every event of every game of a file (JSON lines), for
// the cross-check against python/q4/reference.py in python/tests/test_q4_viewer.py.
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const ctx = vm.createContext({ console });
vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "js", "core.js"), "utf8") + "\nthis.api = { parseGame, pathsAt };", ctx);
for (const line of fs.readFileSync(process.argv[2], "utf8").split(/\r?\n/).filter(l => l.startsWith("(;"))) {
  const g = ctx.api.parseGame(line), out = [];
  for (let k = 0; k <= g.events.length; k++) out.push(Array.from(ctx.api.pathsAt(g, k).dist));
  console.log(JSON.stringify(out));
}
