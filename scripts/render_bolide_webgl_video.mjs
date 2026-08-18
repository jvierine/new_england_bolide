#!/usr/bin/env node

import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const root = path.resolve(import.meta.dirname, "..");
const source = path.join(root, "reports", "figures", "bolide_webgl_volume_viewer.html");
const output = process.argv[2] ?? path.join(root, "products", "web_video", "cape_cod_bay_bolide_4k.mp4");
const width = Number(process.env.VIDEO_WIDTH ?? 3840);
const height = Number(process.env.VIDEO_HEIGHT ?? 2160);
const deviceScaleFactor = Number(process.env.VIDEO_DEVICE_SCALE ?? (width >= 3840 ? 2 : 1));
const fps = Number(process.env.VIDEO_FPS ?? 30);
const duration = Number(process.env.VIDEO_DURATION ?? 26);
const frames = Math.round(fps * duration);

await mkdir(path.dirname(output), { recursive: true });

const browser = await chromium.launch({
  headless: true,
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  args: ["--ignore-gpu-blocklist", "--enable-webgl", "--use-angle=swiftshader"],
});
const viewportWidth = Math.round(width / deviceScaleFactor);
const viewportHeight = Math.round(height / deviceScaleFactor);
const page = await browser.newPage({
  viewport: { width: viewportWidth, height: viewportHeight },
  deviceScaleFactor,
});
await page.goto(pathToFileURL(source).href, { waitUntil: "load" });
await page.waitForFunction(() => typeof frame === "function" && typeof SolarApproachScene !== "undefined");
await page.evaluate(() => {
  playing = false;
  document.getElementById("play").textContent = "Play";
  document.getElementById("play").classList.remove("active");
});

const ffmpeg = spawn(
  "ffmpeg",
  [
    "-hide_banner",
    "-loglevel", "warning",
    "-y",
    "-f", "image2pipe",
    "-framerate", String(fps),
    "-vcodec", "mjpeg",
    "-i", "pipe:0",
    "-an",
    "-c:v", "libx264",
    "-preset", "slow",
    "-crf", "16",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    output,
  ],
  { stdio: ["pipe", "inherit", "inherit"] },
);

for (let index = 0; index < frames; index += 1) {
  const progress = index / frames;
  await page.evaluate((value) => {
    playProgress = value;
    simT = progressToTime(atmosphereProgress(playProgress));
  }, progress);
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const jpeg = await page.screenshot({ type: "jpeg", quality: 97 });
  if (!ffmpeg.stdin.write(jpeg)) await once(ffmpeg.stdin, "drain");
  if ((index + 1) % fps === 0 || index + 1 === frames) {
    process.stdout.write(`Rendered ${index + 1}/${frames} frames (${((index + 1) / fps).toFixed(0)} s)\n`);
  }
}

ffmpeg.stdin.end();
const [exitCode] = await once(ffmpeg, "close");
await browser.close();
if (exitCode !== 0) throw new Error(`ffmpeg exited with status ${exitCode}`);
process.stdout.write(`${output}\n`);
