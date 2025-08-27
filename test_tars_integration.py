import os
import playsound

def test_tars_audio():
    voice_dir = "voice_cloning/generated_voices"
    files = ["tars_humor.wav", "tars_status.wav", "tars_notification.wav"]
    
    print("🤖 Testing TARS voice files...")
    
    for file in files:
        path = os.path.join(voice_dir, file)
        if os.path.exists(path):
            print(f"🎤 Playing: {file}")
            playsound.playsound(path)
            input("Press Enter for next...")
        else:
            print(f"❌ File not found: {path}")
    
    print("✅ TARS voice test complete!")

if __name__ == "__main__":
    test_tars_audio()
