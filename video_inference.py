"""
Video Inference Script for Car Retrieval System.

Processes video files to detect and classify cars using the trained pipeline.

Usage:
    python video_inference.py --input traffic_test.mp4 --output output_video.mp4
"""

import os
import sys
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import CarRetrievalPipeline, create_pipeline
from utils import ensure_dir, create_video_writer


def get_args():
    parser = argparse.ArgumentParser(description='Video Inference for Car Retrieval')
    
    parser.add_argument('--input', '-i', type=str, default='traffic_test.mp4',
                        help='Input video path')
    parser.add_argument('--output', '-o', type=str, default='outputs/output_video.mp4',
                        help='Output video path')
    parser.add_argument('--classifier_model', type=str, default=None,
                        help='Path to trained classifier model')
    parser.add_argument('--detector_confidence', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='Process every N frames (1 = all frames)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process (None = all)')
    parser.add_argument('--show_fps', action='store_true', default=True,
                        help='Show FPS on output video')
    parser.add_argument('--show_stats', action='store_true', default=True,
                        help='Show detection stats on output video')
    parser.add_argument('--save_frames', action='store_true', default=False,
                        help='Save individual frames as images')
    
    return parser.parse_args()


def draw_stats(
    frame: np.ndarray,
    stats: dict,
    fps: float
) -> np.ndarray:
    """Draw statistics overlay on frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    
    # Draw stats
    y_offset = 35
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), font, font_scale, color, thickness)
    
    y_offset += 25
    cv2.putText(frame, f"Cars Detected: {stats.get('num_cars', 0)}", 
                (20, y_offset), font, font_scale, color, thickness)
    
    y_offset += 25
    cv2.putText(frame, f"Frame: {stats.get('frame_num', 0)}/{stats.get('total_frames', 0)}", 
                (20, y_offset), font, font_scale, color, thickness)
    
    y_offset += 25
    cv2.putText(frame, f"Inference: {stats.get('inference_time', 0)*1000:.1f}ms", 
                (20, y_offset), font, font_scale, color, thickness)
    
    return frame


def process_video(args):
    """Main video processing function."""
    
    # Check input exists
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return
    
    # Create output directory
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    
    # Open video
    print(f"Opening video: {args.input}")
    cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f}s")
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = create_pipeline(
        classifier_model_path=args.classifier_model,
        detector_confidence=args.detector_confidence
    )
    
    # Create video writer
    output_fps = fps / args.skip_frames
    writer = create_video_writer(str(output_path), output_fps, (width, height))
    
    # Create frames directory if saving frames
    if args.save_frames:
        frames_dir = output_path.parent / 'frames'
        ensure_dir(frames_dir)
    
    # Processing statistics
    total_cars_detected = 0
    car_type_counts = {}
    processing_times = []
    
    # Process frames
    print(f"\nProcessing video...")
    frame_num = 0
    processed_frames = 0
    
    pbar = tqdm(total=min(total_frames, args.max_frames or total_frames))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Check max frames
        if args.max_frames and frame_num > args.max_frames:
            break
        
        # Skip frames
        if frame_num % args.skip_frames != 0:
            continue
        
        # Process frame
        start_time = time.time()
        result = pipeline.process(frame, return_visualization=True)
        inference_time = time.time() - start_time
        processing_times.append(inference_time)
        
        # Update statistics
        num_cars = result['num_cars']
        total_cars_detected += num_cars
        
        for det in result['detections']:
            car_type = det.get('car_type', 'unknown')
            car_type_counts[car_type] = car_type_counts.get(car_type, 0) + 1
        
        # Get visualization
        vis_frame = result.get('visualization', frame)
        
        # Add stats overlay
        if args.show_fps or args.show_stats:
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            stats = {
                'num_cars': num_cars,
                'frame_num': frame_num,
                'total_frames': total_frames,
                'inference_time': inference_time
            }
            vis_frame = draw_stats(vis_frame, stats, current_fps)
        
        # Write frame
        writer.write(vis_frame)
        processed_frames += 1
        
        # Save individual frame
        if args.save_frames:
            frame_path = frames_dir / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(frame_path), vis_frame)
        
        pbar.update(1)
        pbar.set_postfix({
            'cars': num_cars,
            'fps': f'{1.0/inference_time:.1f}'
        })
    
    pbar.close()
    
    # Cleanup
    cap.release()
    writer.release()
    
    # Print summary
    avg_time = np.mean(processing_times) if processing_times else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nOutput saved to: {output_path}")
    print(f"\nProcessing Summary:")
    print(f"  Frames processed: {processed_frames}")
    print(f"  Average inference time: {avg_time*1000:.1f}ms")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Total car detections: {total_cars_detected}")
    
    print(f"\nCar Type Distribution:")
    for car_type, count in sorted(car_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {car_type}: {count}")
    
    # Save statistics
    stats_path = output_path.parent / 'video_stats.txt'
    with open(stats_path, 'w') as f:
        f.write("Video Inference Statistics\n")
        f.write("="*40 + "\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Output: {output_path}\n\n")
        f.write(f"Frames processed: {processed_frames}\n")
        f.write(f"Average inference time: {avg_time*1000:.1f}ms\n")
        f.write(f"Average FPS: {avg_fps:.1f}\n")
        f.write(f"Total car detections: {total_cars_detected}\n\n")
        f.write("Car Type Distribution:\n")
        for car_type, count in sorted(car_type_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {car_type}: {count}\n")
    
    print(f"\nStatistics saved to: {stats_path}")


def main():
    args = get_args()
    process_video(args)


if __name__ == '__main__':
    main()
