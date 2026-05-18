const childProcess = require("child_process");
const fs = require("fs");
const path = require("path");

const npmRoot = childProcess.execSync("npm root -g", { encoding: "utf8" }).trim();
const distDir = path.join(npmRoot, "openclaw", "dist");
const matches = fs
  .readdirSync(distDir)
  .filter((name) => /^selection-.*\.js$/.test(name));

if (matches.length < 1) {
  throw new Error("Expected at least one selection bundle");
}

const original =
  "const clampImplicitTimeoutMs = (valueMs) => clampTimeoutMs(Math.min(valueMs, DEFAULT_LLM_IDLE_TIMEOUT_MS));";
const patched =
  "const clampImplicitTimeoutMs = (valueMs) => clampTimeoutMs(valueMs);";

let patchedCount = 0;
let alreadyPatchedCount = 0;
for (const match of matches) {
  const bundlePath = path.join(distDir, match);
  let source = fs.readFileSync(bundlePath, "utf8");
  if (source.includes(patched)) {
    alreadyPatchedCount += 1;
    continue;
  }
  if (!source.includes(original)) {
    continue;
  }
  source = source.replace(original, patched);
  fs.writeFileSync(bundlePath, source);
  patchedCount += 1;
  console.log(`Patched OpenClaw LLM idle timeout cap: ${bundlePath}`);
}

if (patchedCount + alreadyPatchedCount < 1) {
  throw new Error(`OpenClaw idle-timeout pattern not found in selection bundles: ${matches.join(", ")}`);
}

const piBundles = fs
  .readdirSync(distDir)
  .filter((name) => /^pi-embedded-.*\.js$/.test(name));
const retryOriginal = "const MAX_SAME_MODEL_IDLE_TIMEOUT_RETRIES = 1;";
const retryPatched = "const MAX_SAME_MODEL_IDLE_TIMEOUT_RETRIES = 0;";
let retryPatchedCount = 0;
let retryAlreadyPatchedCount = 0;

for (const match of piBundles) {
  const bundlePath = path.join(distDir, match);
  let source = fs.readFileSync(bundlePath, "utf8");
  if (source.includes(retryPatched)) {
    retryAlreadyPatchedCount += 1;
    continue;
  }
  if (!source.includes(retryOriginal)) {
    continue;
  }
  source = source.replace(retryOriginal, retryPatched);
  fs.writeFileSync(bundlePath, source);
  retryPatchedCount += 1;
  console.log(`Patched OpenClaw same-model idle retries: ${bundlePath}`);
}

if (retryPatchedCount + retryAlreadyPatchedCount < 1) {
  throw new Error(`OpenClaw same-model idle retry pattern not found in pi bundles: ${piBundles.join(", ")}`);
}
