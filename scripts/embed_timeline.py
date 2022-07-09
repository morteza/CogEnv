# %%

# AndroidEnv requires content of the ADB push to be embedded into the textproto.
# This script load the behaverse_bm.textproto and embeds the content of the BM.json into
# its adb_request.push.content filed.

import json
from google.protobuf import text_format

from android_env.proto import task_pb2

task_proto_path = 'cogenv/proto/belval_matrices.textproto'
timeline_json_path = 'vendor/BM.json'
output_proto_path = 'vendor/BM.textproto'

task = task_pb2.Task()

# parse input
with open(task_proto_path, 'r') as proto_file:
    text_format.Parse(proto_file.read(), task)  # type: ignore

# add content to the push message
with open(timeline_json_path, 'rb') as f:
    content = json.loads(f.read())
    content_minified = json.dumps(content, separators=(',', ':')).encode('utf-8')

    for i, step in enumerate(task.setup_steps):
        if 'adb_request' in str(step) and 'push' in str(step):
            task.setup_steps[i].adb_request.push.content = content_minified  # type: ignore

# write output
with open(output_proto_path, 'w') as f:
    f.write(text_format.MessageToString(task))  # type: ignore

print('Saved the proto to {}.'.format(output_proto_path))
