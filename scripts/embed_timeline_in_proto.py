# %%

# AndroidEnv requires content of the ADB push to be embedded into the textproto.
# This script load the behaverse_bm.textproto and embeds the content of the BM.json into
# its adb_request.push.content filed.

from google.protobuf import text_format

from android_env.proto import task_pb2

task_proto_path = 'cog_env/proto/behaverse_bm.textproto'
timeline_json_path = 'vendor/BM.json'
output_proto_path = 'vendor/BM.textproto'

task = task_pb2.Task()

# parse input
with open(task_proto_path, 'r') as proto_file:
    text_format.Parse(proto_file.read(), task)  # type: ignore

# add content to the push message
with open(timeline_json_path, 'rb') as f:
    timeline_content = f.read()
    task.setup_steps[1].adb_request.push.content = timeline_content  # type: ignore

# write output
with open(output_proto_path, 'w') as f:
    f.write(text_format.MessageToString(task))  # type: ignore
