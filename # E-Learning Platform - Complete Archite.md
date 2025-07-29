# E-Learning Platform - Complete Architecture & Development Plan

## 🏗 Overall Architecture

### Frontend Stack (Angular 20+)
- **Framework**: Angular 20+ with standalone components
- **UI Framework**: PrimeNg + Tailwind CSS
- **State Management**: Angular Signals
- **Routing**: Angular Router with guards
- **Forms**: Angular Reactive Forms
- **HTTP Client**: Angular HttpClient with interceptors
- **Authentication**: JWT with refresh tokens
- **Video Player**: Video.js
- **Charts**: Chart.js
- **Markdown Processing**: Marked.js for markdown parsing and rendering
- **File Upload**: primeng
- **Notifications**: PrimeNg toast + Push notifications
- **PWA**: Angular Service Worker
- **Testing**: Playwright

### Backend Stack (FastAPI)
- **Framework**: FastAPI with Python 3.11+
- **Database**: PostgreSQL with asyncpg
- **ORM**: SQLAlchemy 2.0 (async)
- **Authentication**: JWT with passlib + OAuth2 (Google, Microsoft, GitHub)
- **OAuth Libraries**: authlib
- **File Storage**: Azure Blob Storage
- **Caching**: Redis
- **Task Queue**: Celery with Redis broker
- **Email**: Mailchimp
- **Video Processing**: FFmpeg
- **Search**: Elasticsearch (optional for advanced search)
- **Monitoring**: Prometheus + Grafana
- **Documentation**: FastAPI automatic OpenAPI
- **Testing**: pytest + pytest-asyncio

### Infrastructure & DevOps
- **Containerization**: Docker + Docker Compose
- **Cloud**: Azure (App Service, Database, Blob Storage)
- **CI/CD**: GitHub Actions or Azure DevOps
- **Reverse Proxy**: Nginx
- **SSL**: Let's Encrypt
- **Monitoring**: Application Insights

## 📊 Database Schema Design

### Core Tables
```sql
-- Users and Authentication
users (id, email, password_hash, role, is_active, email_verified, timezone, login_count, last_login, marketing_emails,terms_accepted, terms_accepted_at,  suspended, suspended_at, suspended_reason, deletion_requested, deletion_requested_at, created_at, updated_at)
user_profiles (user_id, first_name, last_name, bio, profession, photo_url, objectives, phone, website, linkedin_url, github_url, country, city, postal_code, company, job_title, experience_level, years_experience, industry, created_at, updated_at)
user_sessions (id, user_id, token, expires_at, created_at, is_suspicious, requires_verification, verified_at)
oauth_accounts (id, user_id, provider, provider_user_id, access_token, refresh_token, created_at)

-- Courses and Content
courses (id, title, description, level, duration, certification, created_by, is_active, category, subcategory, language, price, discount_price, thumbnail_url, trailer_video_url, prerequisites, learning_objectives, target_audience, tags, enrollment_count, average_rating, review_count, completion_rate, difficulty_score, estimated_effort, created_at, updated_at, published_at, free_preview)
course_modules (id, course_id, title, description, order_index, is_active, duration_minutes, learning_objectives, is_free_preview, prerequisite_modules, difficulty_level, module_type, estimated_effort, completion_criteria)
lessons (id, module_id, title, content, video_url, resources, order_index, duration_minutes, is_published, is_free_preview, created_at, updated_at)
quizzes (id, lesson_id, title, questions, passing_score, shuffle_questions, shuffle_answers, created_at, updated_at)
assignments (id, lesson_id, title, description, max_score, due_date, submission_format, allowed_file_types, is_active, created_at, updated_at)

-- Progress and Analytics
user_course_enrollments (user_id, course_id, enrolled_at, completed_at, progress_percentage, enrollment_type, payment_status, payment_amount, currency, discount_applied, coupon_code, current_lesson_id, last_accessed, modules_completed, lessons_completed, quizzes_completed, assignments_completed, is_favorite, certificate_issued, certificate_issued_at, certificate_url, average_quiz_score, average_assignment_score, course_rating, course_review)
lesson_progress (user_id, lesson_id, completed_at, time_spent, status, progress_percentage, first_accessed, last_accessed, video_progress, video_completed)
quiz_attempts (id, user_id, quiz_id, score, answers, attempted_at)
assignment_submissions (id, user_id, assignment_id, content, files, submitted_at, score, feedback)

-- Gamification
user_xp (user_id, total_xp, daily_streak, last_activity)
badges (id, name, description, criteria, icon_url)
user_badges (user_id, badge_id, earned_at)

-- Mentoring
mentor_assignments (mentor_id, student_id, assigned_at)
appointments (id, mentor_id, student_id, scheduled_at, duration, meeting_url, status)
chat_messages (id, sender_id, receiver_id, message, sent_at, read_at)

-- Community
forum_topics (id, course_id, user_id, title, content, created_at)
forum_replies (id, topic_id, user_id, content, created_at, votes)
```

## 🎨 Frontend Architecture

### Project Structure
```
src/
├── app/
│   ├── core/                    # Singleton services, guards, interceptors
│   │   ├── guards/
│   │   ├── interceptors/
│   │   ├── services/
│   │   └── models/
│   ├── shared/                  # Reusable components, directives, pipes
│   │   ├── components/
│   │   ├── directives/
│   │   ├── pipes/
│   │   └── utils/
│   ├── features/                # Feature modules
│   │   ├── auth/
│   │   ├── dashboard/
│   │   ├── courses/
│   │   ├── learning/
│   │   ├── mentoring/
│   │   ├── profile/
│   │   ├── admin/
│   │   └── community/
│   ├── layout/                  # Layout components
│   └── app.component.ts
├── assets/
├── environments/
└── styles/
    ├── tailwind.css
    └── components/
```

### Shared Components & Services

**Markdown Processing:**
- Markdown parser service using Marked.js
- Syntax highlighting with highlight.js
- Custom markdown renderer component
- Markdown editor with live preview
- Support for LaTeX math expressions (optional)

**Content Creation Tools:**
- Markdown-based course content editor
- Live preview for instructors
- Markdown templates for common content types
- File upload integration for images/attachments

### Key Components Architecture

**Authentication Module**
- Login/Register components with email/password
- OAuth integration (Google, Microsoft/Hotmail, GitHub)
- Social login buttons and callbacks
- Password reset for email accounts
- Account linking (merge OAuth with existing accounts)
- Route guards (AuthGuard, RoleGuard)

**Content Management Module**
- Markdown-based content editor
- Live markdown preview
- Syntax highlighting for code blocks
- Image/file upload integration
- Content versioning and drafts

**Dashboard Module**
- Personal dashboard
- Progress tracking
- Upcoming appointments
- Notifications center

**Learning Module**
- Video player with controls
- Markdown content renderer
- Quiz engine
- Assignment submission (supporting markdown)
- Progress tracking
- Bookmark system

**Course Management**
- Course catalog with filters
- Course details (markdown-rendered descriptions)
- Enrollment system
- Recommendation engine

**Mentoring Module**
- Appointment scheduling
- Video conference integration
- Chat system (with markdown support)
- Session history

**Gamification Module**
- XP tracking
- Badge system
- Leaderboards
- Achievement notifications

## ⚙️ Backend Architecture

### Project Structure
```
app/
├── api/                         # API routes
│   ├── v1/
│   │   ├── auth/
│   │   ├── courses/
│   │   ├── users/
│   │   ├── learning/
│   │   ├── mentoring/
│   │   └── admin/
├── core/                        # Core functionality
│   ├── config.py
│   ├── security.py
│   ├── database.py
│   └── deps.py
├── models/                      # SQLAlchemy models
├── schemas/                     # Pydantic schemas
├── services/                    # Business logic
├── tasks/                       # Celery tasks
├── utils/                       # Utilities
└── main.py
```

### Key Services

**Authentication Service**
- JWT token management with refresh tokens
- OAuth2 integration (Google, Microsoft, GitHub)
- Account linking and merging
- Email verification for password accounts
- Password hashing and validation
- Session management and device tracking

**Course Service**
- Course CRUD operations
- Enrollment management
- Progress tracking
- Content delivery

**Learning Service**
- Video streaming
- Quiz processing
- Assignment evaluation
- Certificate generation

**Notification Service**
- Email notifications
- Push notifications
- In-app notifications

**File Service**
- Azure Blob Storage integration
- Video processing
- Secure file access

**Analytics Service**
- User progress analytics
- Course performance metrics
- Engagement tracking


## 🚀 Development Workflow & Phases

### Phase 1: Foundation (Weeks 1-4)
**Priority: Core Infrastructure**

**Backend Tasks:**
1. Setup FastAPI project structure
2. Configure PostgreSQL database with OAuth tables
3. Implement SQLAlchemy models (including oauth_accounts)
4. Setup authentication system (JWT + OAuth2)
5. Configure Google, Microsoft, and GitHub OAuth
6. Create basic CRUD operations
7. Setup Azure Blob Storage
8. Implement OAuth API endpoints
9. User management API

**Frontend Tasks:**
1. Setup Angular 20 project with Tailwind CSS and PrimeNg
2. Configure routing and navigation (including OAuth callbacks)
3. Implement authentication components with OAuth buttons
4. Create OAuth callback handling
5. Create shared UI components
6. Setup state management (Signals)
7. Implement responsive layout
8. User management interface

**Deliverables:**
- Working authentication system (email/password + OAuth)
- OAuth integration with Google, Microsoft, and GitHub
- Account linking and user management
- Project foundation ready

### Phase 2: Core Learning Features (Weeks 5-8)
**Priority: Essential Learning Functionality**

**Backend Tasks:**
1. Basic CMS APIs - Course CRUD, Module CRUD, Lesson CRUD
2. Video streaming endpoints
3. Quiz system API
4. Assignment submission API
5. Progress tracking
6. File upload/download

**Frontend Tasks:**
1. Course catalog with search/filters
2. Video player integration
3. Quiz interface
4. Assignment submission forms
5. Progress tracking dashboard
6. Course enrollment system
7. Basic Content Management Interface - Course creation forms
8. Rich text editor - Markdown editor for lesson content

**Deliverables:**
- Complete course viewing experience
- Working quiz and assignment system
- Progress tracking

### Phase 3: Advanced Features (Weeks 9-12)
**Priority: Enhanced User Experience**

**Backend Tasks:**
1. Advanced CMS features - Quiz builder, Assignment creator
2. Content publishing workflow - Draft/Review/Publish states
3. Mentoring system API
4. Chat/messaging system
5. Notification system
6. Certificate generation

**Frontend Tasks:**
1. Advanced Content Creation Tools - Quiz builder, Assignment creator
2. Content Preview System - See content as students will
3. Mentoring interface
4. Real-time chat
5. Notification system
6. Certificate display

**Deliverables:**
- Complete mentoring system
- Content management tools

### Phase 4: Admin & Analytics (Weeks 13-16)
**Priority: Management and Insights**

**Backend Tasks:**
1. Admin dashboard API
2. Gamification backend
3. Forum/community API
4. Analytics and reporting
5. System monitoring
6. Performance optimization

**Frontend Tasks:**
1. Admin dashboard
2. Advanced Content Creation Tools - Quiz builder, Assignment creator
3. Gamification UI
4. Community forum
5. Analytics dashboard
6. System monitoring UI

**Deliverables:**
- Complete admin system
- Analytics and reporting
- Gamification features
- Community features

### Phase 5: Polish & Deployment (Weeks 17-20)
**Priority: Production Readiness**

**Tasks:**
1. Performance optimization
2. Security hardening
3. Testing and QA
4. Documentation
5. Deployment setup
6. Monitoring and logging

**Deliverables:**
- Production-ready application
- Complete documentation
- Monitoring and alerting setup

```

## 📋 Key Implementation Considerations

### Security
- Implement proper CORS configuration
- Use HTTPS in production
- Sanitize user inputs
- Implement rate limiting
- Secure file uploads
- Use environment variables for secrets

### Performance
- Implement lazy loading for Angular modules
- Use OnPush change detection strategy
- Optimize database queries
- Implement caching strategies
- Use CDN for static assets
- Compress images and videos

### Scalability
- Design stateless APIs
- Use horizontal scaling
- Implement database indexing
- Use async operations
- Implement proper error handling
- Monitor performance metrics

### User Experience
- Implement offline capabilities (PWA)
- Add loading states and skeletons
- Ensure mobile responsiveness
- Implement proper error messages
- Add keyboard navigation
- Follow accessibility guidelines
