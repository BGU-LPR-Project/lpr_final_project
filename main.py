from video_service import VideoService
from edge_service import EdgeService
from cloud_service import CloudService
import time

video = VideoService("recordings/motion4.mp4")
edge = EdgeService()
cloud = CloudService()

video.on()
edge.on()
cloud.on()

try:
    while True:
        try:
            item = video.predict(None, None)
            if item:
                edge.input_queue.put(item)
        except:
            pass
        try:
            item = edge.output_queue.get(timeout=0.1)
            cloud.input_queue.put(item)
            edge.output_queue.task_done()
        except:
            pass
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    video.off()
    edge.off()
    cloud.off()