from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

def create_presentation():
    prs = Presentation()
    
    # Solarized Light Palette
    COLORS = {
        'base3': RGBColor(253, 246, 227),  # #fdf6e3 Background
        'base2': RGBColor(238, 232, 213),  # #eee8d5
        'base1': RGBColor(147, 161, 161),  # #93a1a1
        'base00': RGBColor(101, 123, 131), # #657b83 Text
        'base01': RGBColor(88, 110, 117),  # #586e75
        'yellow': RGBColor(181, 137, 0),   # #b58900
        'orange': RGBColor(203, 75, 22),   # #cb4b16
        'red':    RGBColor(220, 50, 47),   # #dc322f
        'magenta': RGBColor(211, 54, 130), # #d33682
        'violet': RGBColor(108, 113, 196), # #6c71c4
        'blue':   RGBColor(38, 139, 210),  # #268bd2
        'cyan':   RGBColor(42, 161, 152),  # #2aa198
        'green':  RGBColor(133, 153, 0),   # #859900
    }

    def set_background(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = COLORS['base3']

    def add_title_slide(title_text, subtitle_text):
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        set_background(slide)
        
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        
        title.text = title_text
        subtitle.text = subtitle_text
        
        # Style Title
        title_tf = title.text_frame
        title_tf.paragraphs[0].font.color.rgb = COLORS['orange']
        title_tf.paragraphs[0].font.size = Pt(54)
        title_tf.paragraphs[0].font.bold = True
        
        # Style Subtitle
        subtitle_tf = subtitle.text_frame
        p = subtitle_tf.paragraphs[0]
        p.font.color.rgb = COLORS['base01']
        p.font.size = Pt(28)
        
        return slide

    def add_content_slide(title_text, content_items):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        set_background(slide)
        
        title = slide.shapes.title
        title.text = title_text
        title.text_frame.paragraphs[0].font.color.rgb = COLORS['yellow']
        title.text_frame.paragraphs[0].font.bold = True
        
        body = slide.placeholders[1]
        tf = body.text_frame
        tf.clear()
        
        for item in content_items:
            p = tf.add_paragraph()
            p.text = item
            p.font.color.rgb = COLORS['base00']
            p.font.size = Pt(24)
            p.space_after = Pt(20)

    def add_image_slide(title_text, image_path, caption=None):
        slide = prs.slides.add_slide(prs.slide_layouts[5]) # Title Only
        set_background(slide)
        
        title = slide.shapes.title
        title.text = title_text
        title.text_frame.paragraphs[0].font.color.rgb = COLORS['yellow']
        
        if os.path.exists(image_path):
            img = slide.shapes.add_picture(image_path, Inches(1), Inches(2.0), height=Inches(4.5))
            # Center image
            img.left = int((prs.slide_width - img.width) / 2)
            
            if caption:
                txBox = slide.shapes.add_textbox(Inches(1), Inches(6.6), Inches(8), Inches(1))
                tf = txBox.text_frame
                p = tf.add_paragraph()
                p.text = caption
                p.alignment = PP_ALIGN.CENTER
                p.font.color.rgb = COLORS['base01']
                p.font.size = Pt(16)
        else:
            print(f"Warning: Image not found at {image_path}")
            
    def add_two_column_slide(title_text, col1_title, col1_items, col2_title, col2_items):
        slide = prs.slides.add_slide(prs.slide_layouts[1]) 
        set_background(slide)
        
        title = slide.shapes.title
        title.text = title_text
        title.text_frame.paragraphs[0].font.color.rgb = COLORS['yellow']
        title.text_frame.paragraphs[0].font.bold = True
        
        left = Inches(0.5)
        top = Inches(2.0)
        width = Inches(4.5)
        height = Inches(5.0)
        
        # Column 1
        txBox1 = slide.shapes.add_textbox(left, top, width, height)
        tf1 = txBox1.text_frame
        p = tf1.add_paragraph()
        p.text = col1_title
        p.font.bold = True
        p.font.size = Pt(28)
        p.font.color.rgb = COLORS['blue']
        p.space_after = Pt(20)
        
        for item in col1_items:
            p = tf1.add_paragraph()
            p.text = "• " + item
            p.font.size = Pt(20)
            p.font.color.rgb = COLORS['base00']
            p.level = 0
            p.space_after = Pt(10)
            
        # Column 2
        txBox2 = slide.shapes.add_textbox(left + Inches(5.0), top, width, height)
        tf2 = txBox2.text_frame
        p = tf2.add_paragraph()
        p.text = col2_title
        p.font.bold = True
        p.font.size = Pt(28)
        p.font.color.rgb = COLORS['blue']
        p.space_after = Pt(20)
        
        for item in col2_items:
            p = tf2.add_paragraph()
            p.text = "• " + item
            p.font.size = Pt(20)
            p.font.color.rgb = COLORS['base00']
            p.level = 0
            p.space_after = Pt(10)

    def add_stats_slide(title_text, stats):
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        set_background(slide)
        
        title = slide.shapes.title
        title.text = title_text
        title.text_frame.paragraphs[0].font.color.rgb = COLORS['yellow']
        title.text_frame.paragraphs[0].font.bold = True
        
        start_x = Inches(0.5)
        start_y = Inches(2.5)
        box_width = Inches(2.2)
        box_height = Inches(1.5)
        gap = Inches(0.25)
        
        for i, (value, label) in enumerate(stats):
            left = start_x + (i * (box_width + gap))
            
            # Draw box
            shape = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, left, start_y, box_width, box_height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = COLORS['base2']
            shape.line.color.rgb = COLORS['base1']
            
            txt_box = slide.shapes.add_textbox(left, start_y + Inches(0.25), box_width, Inches(0.8))
            p = txt_box.text_frame.paragraphs[0]
            p.text = value
            p.alignment = PP_ALIGN.CENTER
            p.font.bold = True
            p.font.size = Pt(32)
            p.font.color.rgb = COLORS['magenta']
            
            label_box = slide.shapes.add_textbox(left, start_y + Inches(0.85), box_width, Inches(0.5))
            p = label_box.text_frame.paragraphs[0]
            p.text = label
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(14)
            p.font.color.rgb = COLORS['base01']

    # --- SLIDES ---

    # 1. Title
    add_title_slide(
        "Car Retrieval System",
        "An End-to-End Deep Learning Approach for\nIndonesian Vehicle Detection and Classification"
    )

    # 2. Problem Statement
    add_two_column_slide(
        "Problem Statement",
        "Measurements", [
            "Detect multiple car instances",
            "Classify 8 Indonesian car types",
            "Process traffic video"
        ],
        "Constraints", [
            "Use Deep Learning (PyTorch)",
            "Real-time capability desirable",
            "Robust to varied visual conditions"
        ]
    )

    # 3. System Architecture
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_background(slide)
    title = slide.shapes.title
    title.text = "System Architecture"
    title.text_frame.paragraphs[0].font.color.rgb = COLORS['yellow']
    title.text_frame.paragraphs[0].font.bold = True

    # Draw Flowchart Boxes
    shapes = slide.shapes
    mid_y = Inches(3.5)
    
    # Coordinates for 5 steps
    steps = [
        ("Input\nVideo/Image", COLORS['base1']),
        ("YOLOv8\nDetector", COLORS['blue']),
        ("Crop\nRegions", COLORS['base1']),
        ("ResNet50\nClassifier", COLORS['blue']),
        ("Output\nAnnnotations", COLORS['green'])
    ]
    
    box_w = Inches(1.6)
    box_h = Inches(1.2)
    gap = Inches(0.3)
    start_x = (prs.slide_width - (5 * box_w) - (4 * gap)) / 2
    
    for i, (text, color) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        shape = shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, mid_y, box_w, box_h)
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.color.rgb = COLORS['base01']
        
        tf = shape.text_frame
        tf.clear()
        p = tf.add_paragraph()
        p.text = text
        p.alignment = PP_ALIGN.CENTER
        p.font.color.rgb = RGBColor(255, 255, 255) # White text for contrast
        p.font.bold = True
        
        # Arrow
        if i < len(steps) - 1:
            arrow = shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + box_w + Inches(0.05), mid_y + box_h/2 - Inches(0.1), Inches(0.2), Inches(0.2))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = COLORS['orange']
            arrow.line.fill.background()

    # 4. Dataset
    add_stats_slide("Dataset Overview", [
        ("17,171", "Total Images"),
        ("8", "Car Types"),
        ("13,542", "Train Samples"),
        ("1,756", "Test Samples")
    ])
    
    # 5. Classifier Details
    add_content_slide("Technical Implementation", [
        "Detector: YOLOv8-nano (COCO pretrained)",
        "Classifier: ResNet50 (ImageNet pretrained)",
        "Optimizer: AdamW (LR: 1e-4) + Cosine Annealing",
        "Augmentation: RandomCrop, ColorJitter, Rotation",
        "Loss: CrossEntropy with Label Smoothing (0.1)"
    ])

    # 6. Results Summary
    add_stats_slide("Performance Summary", [
        ("86.33%", "Test Accuracy"),
        ("86.53%", "Precision"),
        ("86.34%", "F1-Score"),
        ("13.6", "Video FPS")
    ])

    # 7. Confusion Matrix (Image)
    add_image_slide(
        "Confusion Matrix Analysis", 
        "outputs/evaluation/confusion_matrix.png",
        "Diagonal dominance indicates strong performance. Confusion visible between Crossover/MPV."
    )

    # 8. Per-Class Metrics (Image)
    add_image_slide(
        "Per-Class Performance Metrics", 
        "outputs/evaluation/per_class_metrics.png",
        "Pickup and Truck achieve highest F1-scores (>0.90). Crossover is lowest (0.83)."
    )

    # 9. Video Inference
    add_stats_slide("Video Inference Statistics", [
        ("2,960", "Frames Processed"),
        ("73.7ms", "Latency/Frame"),
        ("14,107", "Detections"),
        ("197s", "Duration")
    ])

    # 10. Conclusion
    add_two_column_slide(
        "Conclusion & Future Work",
        "Key Achievements", [
            "Successful 2-stage pipeline integration",
            "Reliable classification (86%+ accuracy)",
            "Real-time viable processing speed"
        ],
        "Future Improvements", [
            "Adopt Vision Transformers (ViT)",
            "Expand dataset for rare classes",
            "Optimize for Edge Devices (TensorRT)"
        ]
    )

    prs.save('presentation.pptx')
    print("Enhanced presentation saved as presentation.pptx")

if __name__ == "__main__":
    create_presentation()
