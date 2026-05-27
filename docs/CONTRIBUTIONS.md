# Contributions

This project was developed as a graduation project at Kadir Has University 
by a multidisciplinary team of Computer Engineering and Electrical Engineering students.

## Metehan Kılıçlı (Computer Engineering)

### Cloud Infrastructure
- Designed and deployed the entire AWS backend architecture
- AWS Lambda + API Gateway (serverless FastAPI deployment, eu-north-1)
- AWS RDS PostgreSQL database setup and schema design
- AWS S3 bucket for image storage
- AWS Kinesis Video Streams for live caregiver monitoring (HLS)
- AWS SNS email alert system with subscription filter policies
- AWS EventBridge scheduled triggers for reminders and nightly risk scoring

### Backend REST API
- FastAPI application with full CRUD endpoints
- Patient, medication, schedule, and dispensing log management
- Role-based access control (caregiver / patient)

### Database Architecture
- Hybrid SQLite (local) + RDS PostgreSQL (cloud) schema design
- Bidirectional sync engine (30-second interval)
- KVKK-compliant local storage for biometric face vectors

### Face Authentication System (Raspberry Pi)
- Multi-modal liveness detection pipeline
- Eye Aspect Ratio (EAR) blink detection
- Mouth openness ratio (MAR) check
- Head rotation tracking via nose X-coordinate
- face_recognition 128-dimensional embedding storage and comparison

### Swallow Detection Module
- Hand-to-mouth gesture detection via MediaPipe
- Mouth open/close sequence tracking
- YOLOv8 nano object detection for glass/cup near mouth
- Intake verification logging to AWS RDS

### Mobile App (Partial)
- Analytics screen UI implementation in Flutter

---

## Yiğit Keser (Computer Engineering)
- Flutter mobile application development
- Caregiver and patient dashboards
- Push notification integration (FCM)

## Bengisu Çağan (Electrical Engineering)
- Hardware design and electronics
- Motor control circuits
- Power system design

## Doğukan Manav (Electrical Engineering)
- Mechanical design (SolidWorks)
- 3D printed carousel and enclosure
- Servo and stepper motor integration
