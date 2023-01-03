Launch CogEnv emulator like this: 

```
tmp/emulator/emulator -no-window -no-snapshot -gpu swiftshader_indirect -no-audio -show-kernel -verbose -avd CogEnv -grpc 17482 -ports 18609,23979
```

And then use the following command to verify:

```
/tmp/platform-tools/adb devices
```
