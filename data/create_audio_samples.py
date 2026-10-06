import subprocess
import os

os.makedirs('data/demo_samples', exist_ok=True)
out_file = os.path.abspath('data/demo_samples/ai_synthetic_speech.wav')

ps_script = f'''
$s = New-Object -ComObject SAPI.SpVoice
$stream = New-Object -ComObject SAPI.SpFileStream
$stream.Open('{out_file}', 3)
$s.AudioOutputStream = $stream
$s.Speak('Shocking secret leaked! The government has secretly replaced all public currency starting tonight.')
$stream.Close()
'''

res = subprocess.run(["powershell", "-Command", ps_script], capture_output=True, text=True)
if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
    print(f"SUCCESS: Created synthetic speech at {out_file} (Size: {os.path.getsize(out_file)} bytes)")
else:
    print("FAILED:", res.stderr)
