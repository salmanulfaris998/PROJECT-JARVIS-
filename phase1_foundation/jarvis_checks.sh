adb shell screencap -p /sdcard/test.png
adb pull /sdcard/test.png
adb shell pm list packages -3 | wc -l
adb shell rm /sdcard/test.png
