import argparse
from profiler.profiler import Profiler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,
                        choices=['image_classification', 'object_detection', 'sign_recognition', 'sound_classification',
                                 'vehicle_detection', 'face_detection', 'age_classification', 'gender_classification',
                                 'emotion_classification', 'wildfire_detection', 'wildlife_recognition',
                                 'scene_recognition', 'traffic_detection'])
    parser.add_argument('--weights', type=str)
    parser.add_argument('--host', type=str, default='192.168.137.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--buffer', type=int, default=1024 * 1024)
    parser.add_argument('--data_dir', type=str, default=r'C:\Users\lxhan2\data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    p = Profiler(args)
    p.profile_mem()
    p.profile_latency()
    p.profile_accuracy()
    p.print()
    p.save()
