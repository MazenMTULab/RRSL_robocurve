# Simulation setup (from the official walkthrough)

Launchable: https://brev.nvidia.com/launchable/deploy?launchableID=env-3LByDjf93lt4rpb6IrwWORhoYPy
Needs: L40S (or other RTX) GPU, ~200 GB storage, secure port 6080 exposed. Redeem the $200 coupon first (Guide > Compute).

On the instance (`brev login; brev shell <instance>` from the laptop):
1. `tail -f /data/yam-setup.log` until `=== [yam-setup] done`, then Ctrl-C.
2. `inspect-robots list embodiments | grep yam_arena`
3. `mkdir -p ~/.dreamlabs && echo 'DREAM_API_KEY=dl_live_...' > ~/.dreamlabs/credentials && chmod 600 ~/.dreamlabs/credentials`
4. Smoke test Dream-1 (scene `station`) and scripted demo:
   `inspect-robots run --embodiment yam_arena --policy dream -E scene=station -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "put apple into the bowl" --scorer success_at_end --max-steps 2400 --store-frames --log-dir ~/yam-eval/dream`
   same with `--policy yam_scripted --max-steps 600 --log-dir ~/yam-eval/demo`
5. Videos: `for log in ~/yam-eval/dream/*.json; do inspect-robots video "$log"; done`
   Screenshot: `ffmpeg -y -i ~/yam-eval/dream/frames/<run>/scene-0-e0_top_cam.mp4 -frames:v 1 ~/yam-eval/setup.png`
6. Reports: `inspect-robots view ~/yam-eval/dream --serve --force --port 8301`; laptop: `brev port-forward <instance> -p 8301:8301`
7. Live desktop: `cat ~/yam-desktop-password`; `brev port-forward <instance> -p 6080:6080`; open http://127.0.0.1:6080/vnc.html; add `-E headless=false` (first start ~4 min black window).
8. Own task: `cp ~/yam-isaac-lab-arena/examples/stack_cubes.py ~/my_task.py` (objects, success(), score(), demo()); run with `-E task=~/my_task.py --scorer task_score`.
Convert videos for upload: `ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart out.mp4` (<=50 MB).
