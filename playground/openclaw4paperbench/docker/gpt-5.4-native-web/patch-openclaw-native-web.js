const childProcess = require("child_process");
const fs = require("fs");
const path = require("path");

const npmRoot = childProcess.execSync("npm root -g", { encoding: "utf8" }).trim();
const distDir = path.join(npmRoot, "openclaw", "dist");
const matches = fs
  .readdirSync(distDir)
  .filter((name) => /^codex-native-web-search-core-.*\.js$/.test(name));

if (matches.length !== 1) {
  throw new Error(`Expected one codex-native-web-search-core bundle, found ${matches.length}`);
}

const bundlePath = path.join(distDir, matches[0]);
const original =
  'return params.modelProvider === "openai-codex" || params.modelApi === "openai-codex-responses";';
const patched =
  'return params.modelProvider === "openai-codex" || params.modelApi === "openai-codex-responses" || params.modelProvider === "openai" && params.modelApi === "openai-responses";';
let source = fs.readFileSync(bundlePath, "utf8");

if (!source.includes(patched)) {
  if (!source.includes(original)) {
    throw new Error(`OpenClaw native search eligibility pattern not found in ${bundlePath}`);
  }
  source = source.replace(original, patched);
  fs.writeFileSync(bundlePath, source);
}

console.log(`Patched OpenClaw native web search bundle: ${bundlePath}`);
