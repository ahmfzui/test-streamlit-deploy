import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import urllib.request
from PIL import Image
from streamlit_option_menu import option_menu
from pymongo import MongoClient
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from datetime import datetime

# Set page configuration
st.set_page_config(layout="wide")

# Koneksi ke MongoDB
client = MongoClient("mongodb+srv://regaarzula:YlDDs2OYHYOuuLPc@cluster0.nslprzn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['hydroponic_system']
collection = db['temperature_settings']
inputJam = db['clock_settings']

# URL ESP32 CAM
url = 'http://192.168.1.12/cam-hi.jpg'
image_placeholder = st.empty()

# Kelas objek untuk model YOLO
classNames = ["Immature Sawi", "Mature Sawi", "Non-Sawi", "Partially Mature Sawi", "Rotten"]

def process_frame(frame, model, min_confidence):
    results = model(frame)
    detected_objects = []
    
    for detection in results[0].boxes.data:
        x0, y0 = (int(detection[0]), int(detection[1]))
        x1, y1 = (int(detection[2]), int(detection[3]))
        score = round(float(detection[4]), 2)
        cls = int(detection[5])
        object_name = classNames[cls]
        label = f'{object_name} {score}'

        if score > min_confidence:
            # Draw the bounding box
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # Compute text size
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_width, label_height = label_size
            baseline = max(baseline, 1)  # Ensure baseline is at least 1

            # Define the rectangle background position
            label_x0 = x0
            label_y0 = y1 + 10
            label_x1 = x0 + label_width + 10
            label_y1 = label_y0 + label_height + baseline

            # Draw the filled rectangle as background for the label
            cv2.rectangle(frame, (label_x0, label_y0 - label_height - 10), (label_x1, label_y1), (0, 0, 255), -1)

            # Draw the label text
            cv2.putText(frame, label, (x0 + 5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            detected_objects.append(label)  # Store detected objects

    return frame, detected_objects

def detect_objects_in_image(model, uploaded_file, min_confidence):
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    result_frame, detected_objects = process_frame(image_bgr, model, min_confidence)
    result_image = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption='Processed Image', use_column_width=True)
    
    # Display the last detected label
    if detected_objects:
        last_label = detected_objects[-1]
        st.write(f"Last detected object: {last_label}")
    else:
        st.write("No objects detected.")

def detect_objects_in_video(model, uploaded_file, min_confidence):
    video_file = uploaded_file.read()
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file)
        
    cap = cv2.VideoCapture("temp_video.mp4")
    stframe = st.empty()
    last_detection = st.empty()  # Placeholder for last detected object

    last_detected_label = ""  # Initialize the variable to store the last detected label

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_objects = process_frame(frame, model, min_confidence)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update last detected object status
        if detected_objects:
            last_detected_label = detected_objects[-1]  # Update with new detection
        last_detection.text(f"Last detected object: {last_detected_label}")  # Display last detected label

    cap.release()
    st.write("Video processing completed.")

def main():
    # Sidebar navigation menggunakan option_menu
    menu_selection = option_menu(
        menu_title=None,
        options=["Home", "Monitoring", "Controlling", "Object Detection"],
        icons=["house", "clipboard-data-fill", "gear-wide-connected", "search"],
        default_index=0,
        orientation="horizontal",
    )

    if menu_selection == "Home":
        st.markdown("<h1 style='text-align: center;'>🌱 Hydroponic Tech House - Dashboard 🌱</h1>", unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)
        # Load your image
        logo = Image.open('Hydroponic.jpg')

        # Create three columns
        left_co, cent_co, last_co = st.columns(3)

        # Place the image in the center column
        with cent_co:
            st.image(logo, caption='Hydroponic', use_column_width=False, width=575)

        st.markdown(
            """
            Welcome to Hydroponic Tech House, where you can manage and monitor your hydroponic system smartly!
            Use the sidebar navigation on the left to explore our features.
            """
        )

    elif menu_selection == "Monitoring":
        # Center buttons
        st.markdown("""
            <style>
            .stButton { display: flex; justify-content: center; }
            </style>
            """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<h1 style='text-align: center;'>Monitoring</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>View real-time sensor data and environmental conditions of your hydroponic system.</p>", unsafe_allow_html=True)
            st.markdown("<hr/>", unsafe_allow_html=True)

            # Create a form for Ubidots widgets
            with st.form("monitoring_form"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.subheader("🌡️ Temperature")
                    widget_url1 = "https://stem.ubidots.com/app/dashboards/public/widget/n2DJ6zraCJkvxZYYAQ5egHCTgZLe6E3XBpVtLGnZsoQ"
                    st.components.v1.iframe(widget_url1, width=300, height=300, scrolling=True)

                    st.subheader("🌱 Soil Moisture")
                    widget_url4 = "https://stem.ubidots.com/app/dashboards/public/widget/ST57XPDVjOhWeqD1GHC1ejT2zCuxr078rU-tQH6WNKo"
                    st.components.v1.iframe(widget_url4, width=300, height=300, scrolling=True)

                with col2:
                    st.subheader("🌬️ Air Quality")
                    widget_url2 = "https://stem.ubidots.com/app/dashboards/public/widget/xggZMNEOQq-32hJmgV9ovScYSPNe9iyO77bi1lB5oE4"
                    st.components.v1.iframe(widget_url2, width=300, height=300, scrolling=True)

                    st.subheader("🌬️ Air Quality")
                    widget_url5 = "https://stem.ubidots.com/app/dashboards/public/widget/xggZMNEOQq-32hJmgV9ovScYSPNe9iyO77bi1lB5oE4"
                    st.components.v1.iframe(widget_url5, width=300, height=300, scrolling=True)
                    
                with col3:
                    st.subheader("💧 Humidity")
                    widget_url3 = "https://stem.ubidots.com/app/dashboards/public/widget/XXSQaCPoG41tQ1W33PDj9xphZOO7DwF6tvflxiKnSkE"
                    st.components.v1.iframe(widget_url3, width=300, height=300, scrolling=True)

                    st.subheader("🌬️ Air Quality")
                    widget_url6 = "https://stem.ubidots.com/app/dashboards/public/widget/xggZMNEOQq-32hJmgV9ovScYSPNe9iyO77bi1lB5oE4"
                    st.components.v1.iframe(widget_url6, width=300, height=300, scrolling=True)
                
                with col4:
                    st.subheader("🌬️ Air Quality")
                    widget_url8 = "https://stem.ubidots.com/app/dashboards/public/widget/xggZMNEOQq-32hJmgV9ovScYSPNe9iyO77bi1lB5oE4"
                    st.components.v1.iframe(widget_url8, width=300, height=300, scrolling=True)
                    
                    st.subheader("🌬️ Air Quality")
                    widget_url7 = "https://stem.ubidots.com/app/dashboards/public/widget/xggZMNEOQq-32hJmgV9ovScYSPNe9iyO77bi1lB5oE4"
                    st.components.v1.iframe(widget_url7, width=300, height=300, scrolling=True)
                
                # Add form submit button
                st.write("")
                submit_button = st.form_submit_button('Refresh Widget')

    elif menu_selection == "Controlling":
        st.markdown("<h1 style='text-align: center;'>Controlling</h1>", unsafe_allow_html=True)
        st.markdown("""
            <p style='text-align: center;'>
                Optimize your hydroponic tech house with advanced automation. 
                Control fans and heating lamps based on temperature sensor readings to maintain ideal climate conditions. 
                Additionally, manage the water pump to efficiently irrigate and circulate water. 
                Achieve precision and flexibility in your hydroponic system with our automated controls.
            </p>
        """, unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)

        # Create tabs for controlling temperature and water motor
        tab1, tab2 = st.tabs(["🌡️ Temperature Control", "💦 Water Motor Control"])

        # Center buttons
        st.markdown("""
            <style>
            .stButton { display: flex; justify-content: center; }
            </style>
            """, unsafe_allow_html=True)

        # Temperature Control tab
        with tab1:
            st.write("")
            st.markdown("<h3 style='text-align: center;'>🌡️ Set Water Motor Timing</h3>", unsafe_allow_html=True)
            st.write("")

            with st.form(key='temperature_form'):
                min_temp = st.number_input('Minimum Temperature (°C)', value=20.0, step=0.1)
                max_temp = st.number_input('Maximum Temperature (°C)', value=30.0, step=0.1)
                submit_button = st.form_submit_button(label='Save Temperature Settings')

            if submit_button:
                collection.insert_one({'min_temp': min_temp, 'max_temp': max_temp})
                st.success('Temperature settings have been saved!')

            latest_setting = collection.find().sort([('_id', -1)]).limit(1)
            for setting in latest_setting:
                st.write('Latest Minimum Temperature:', setting['min_temp'])
                st.write('Latest Maximum Temperature:', setting['max_temp'])

        # Water Motor Control tab
        with tab2:
            st.write("")
            st.markdown("<h3 style='text-align: center;'>💦 Set Water Motor Timing</h3>", unsafe_allow_html=True)
            st.write("")

            def validate_time(time_str):
                """Validate and convert the time string format."""
                try:
                    # Attempt to parse the string into a datetime.time object
                    return datetime.strptime(time_str, '%H:%M').time()
                except ValueError:
                    # Return None if the format is incorrect
                    return None

            # Form for motor settings
            with st.form(key='motor_form'):
                # Text input for time
                alarm_time_str = st.text_input('Jam Alarm', value=datetime.now().strftime('%H:%M'))

                # Submit button
                submit_button = st.form_submit_button(label='Save Motor Settings')

                if submit_button:
                    # Validate the time format
                    alarm_time = validate_time(alarm_time_str)
                    
                    if alarm_time:
                        # Convert the time to a string in HH:MM format
                        alarm_time_str = alarm_time.strftime('%H:%M')
                        
                        # Insert into the database
                        inputJam.insert_one({'alarm_time': alarm_time_str})
                        st.success('Water motor settings have been saved!')
                    else:
                        st.error('Invalid time format. Please use HH:MM.')

                latest_jam_setting = inputJam.find().sort([('_id', -1)]).limit(1)
                for setting in latest_jam_setting:
                    st.write('Jam Alarm Terbaru:', setting.get('alarm_time', 'N/A'))

    elif menu_selection == "Object Detection":
        st.markdown("<h1 style='text-align: center;'>Object Detection for Sawi Varieties</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Leverage advanced object detection to identify and classify different Sawi varieties. You can either upload images or videos for analysis or stream live video from an ESP32 CAM for real-time detection. Choose the method that best suits your needs for monitoring and analyzing your hydroponic system. </p>", unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True) 

        model = YOLO('trained_pakcoy.pt')

        # Create tabs
        tab1, tab2 = st.tabs(["📷 Upload Image and Video", "📡 Real-time Object Detection"])

        with tab1:
            st.write("")
            st.markdown("<h3 style='text-align: center;'>📷 Upload Image and Video</h3>", unsafe_allow_html=True)
            st.write("")

            with st.form("object_detection_form"):
                st.write("Upload an image or video to detect objects using YOLOv8.")
                uploaded_file = st.file_uploader("Upload image or video", type=['jpg', 'png', 'mp4'], label_visibility='collapsed')
                min_confidence = st.slider('Confidence Score', 0.0, 1.0, 0.2)
                submit_button = st.form_submit_button(label='Submit')

            if uploaded_file is not None and submit_button:
                if uploaded_file.type.startswith('image'):
                    detect_objects_in_image(model, uploaded_file, min_confidence)
                elif uploaded_file.type.startswith('video'):
                    detect_objects_in_video(model, uploaded_file, min_confidence)

        with tab2:
            st.write("")
            st.markdown("<h3 style='text-align: center;'>📡 Real-time Object Detection from ESP32 CAM</h3>", unsafe_allow_html=True)
            st.write("")

            # Create a form for webcam controls
            with st.form("webcam_form"):
                min_confidence = st.slider('Confidence Score', 0.0, 1.0, 0.2)
                submit_button_start = st.form_submit_button('Start Video Stream')
                submit_button_stop = st.form_submit_button('Stop Video Stream')

            # Center buttons
            st.markdown("""
                <style>
                .stButton { display: flex; justify-content: center; }
                </style>
                """, unsafe_allow_html=True)
            
            # Manage webcam stream control
            if submit_button_start:
                # Jalankan deteksi objek secara real-time
                while True:
                    img_resp = urllib.request.urlopen(url)
                    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                    frame = cv2.imdecode(imgnp, -1)
                    result_frame = process_frame(frame, model, min_confidence)
                    frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            if submit_button_stop:
                st.write("Video stream stopped.")

# Jalankan aplikasi Streamlit
if __name__ == '__main__':
    main()