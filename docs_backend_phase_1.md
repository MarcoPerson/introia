# E-Learning Platform API Routes Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
All protected endpoints require `Authorization: Bearer <access_token>` header.

---

## 🔐 Authentication Routes

### Register User
```http
POST /api/v1/auth/register
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "first_name": "John",
  "last_name": "Doe",
  "terms_accepted": true,
  "marketing_emails": true
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": 1,
    "email": "user@example.com",
    "role": "student",
    "email_verified": false
  }
}
```

### Login User
```http
POST /api/v1/auth/login
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": 1,
    "email": "user@example.com",
    "role": "student",
    "email_verified": true
  }
}
```

### Refresh Token
```http
POST /api/v1/auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Verify Email
```http
POST /api/v1/auth/verify-email
```

**Request Body:**
```json
{
  "token": "verification_token_from_email"
}
```

**Response:**
```json
{
  "message": "Email verified successfully",
  "success": true
}
```

### Resend Verification Email
```http
POST /api/v1/auth/resend-verification
```

**Request Body:**
```json
{
  "email": "user@example.com"
}
```

**Response:**
```json
{
  "message": "Verification email sent if account exists",
  "success": true
}
```

### Change Password
```http
POST /api/v1/auth/change-password
```
🔒 **Requires Authentication**

**Request Body:**
```json
{
  "current_password": "oldpassword123",
  "new_password": "newpassword123"
}
```

**Response:**
```json
{
  "message": "Password changed successfully",
  "success": true
}
```

### Logout
```http
POST /api/v1/auth/logout
```
🔒 **Requires Authentication**

**Response:**
```json
{
  "message": "Logged out successfully",
  "success": true
}
```

---

## 🔑 Password Recovery Routes

### Request Password Reset
```http
POST /api/v1/auth/password/request-reset
```

**Request Body:**
```json
{
  "email": "user@example.com"
}
```

**Response:**
```json
{
  "message": "If an account with this email exists, a reset link has been sent",
  "success": true
}
```

### Reset Password
```http
POST /api/v1/auth/password/reset
```

**Request Body:**
```json
{
  "token": "reset_token_from_email",
  "new_password": "newpassword123"
}
```

**Response:**
```json
{
  "message": "Password reset successfully",
  "success": true
}
```

### Validate Reset Token
```http
GET /api/v1/auth/password/validate-token/{token}
```

**Response:**
```json
{
  "valid": true,
  "message": "Token is valid"
}
```

---

## 🌐 OAuth Routes

### Get OAuth Authorization URL
```http
GET /api/v1/auth/oauth/{provider}/authorize
```

**Providers:** `google`, `microsoft`, `github`

**Response:**
```json
{
  "provider": "google",
  "authorization_url": "https://accounts.google.com/o/oauth2/v2/auth?...",
  "state": "random_state_string"
}
```

### OAuth Callback
```http
GET /api/v1/auth/oauth/{provider}/callback?code={code}&state={state}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": 1,
    "email": "user@example.com",
    "role": "student",
    "email_verified": true
  },
  "is_new_user": false
}
```

### Get User OAuth Accounts
```http
GET /api/v1/auth/oauth/accounts
```
🔒 **Requires Authentication**

**Response:**
```json
[
  {
    "id": 1,
    "user_id": 1,
    "provider": "google",
    "provider_user_id": "123456789",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }
]
```

### Unlink OAuth Account
```http
DELETE /api/v1/auth/oauth/accounts/{provider}
```
🔒 **Requires Authentication**

**Response:**
```json
{
  "message": "Google account unlinked successfully"
}
```

---

## 🔗 Account Linking Routes

### Link OAuth Account
```http
POST /api/v1/auth/account/link
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "userpassword123",
  "provider": "google",
  "oauth_code": "oauth_authorization_code",
  "oauth_state": "oauth_state_string"
}
```

**Response:**
```json
{
  "message": "Google account successfully linked",
  "success": true,
  "linked_accounts": [
    {
      "id": 1,
      "user_id": 1,
      "provider": "google",
      "provider_user_id": "123456789",
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### Get Linkable Accounts
```http
GET /api/v1/auth/account/linkable/{email}
```

**Response:**
```json
{
  "email": "user@example.com",
  "has_password": true,
  "linked_accounts": [
    {
      "provider": "google",
      "linked_at": "2024-01-01T00:00:00Z"
    }
  ],
  "linkable_providers": ["microsoft", "github"]
}
```

### Unlink Account
```http
DELETE /api/v1/auth/account/unlink/{provider}
```
🔒 **Requires Authentication**

**Response:**
```json
{
  "message": "Google account unlinked successfully",
  "success": true
}
```

### Get Account Linking Status
```http
GET /api/v1/auth/account/status
```
🔒 **Requires Authentication**

**Response:**
```json
{
  "email": "user@example.com",
  "has_password": true,
  "linked_accounts": [
    {
      "provider": "google",
      "linked_at": "2024-01-01T00:00:00Z"
    }
  ],
  "linkable_providers": ["microsoft", "github"]
}
```

---

## 👤 User Management Routes

### Get Current User Profile
```http
GET /api/v1/users/me
```
🔒 **Requires Authentication**

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "role": "student",
  "is_active": true,
  "email_verified": true,
  "login_count": 5,
  "last_login": "2024-01-01T12:00:00Z",
  "terms_accepted": true,
  "terms_accepted_at": "2024-01-01T00:00:00Z",
  "suspended": false,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "profile": {
    "user_id": 1,
    "first_name": "John",
    "last_name": "Doe",
    "bio": "Software developer",
    "profession": "Developer",
    "photo_url": "https://example.com/photo.jpg",
    "phone": "+1234567890",
    "website": "https://johndoe.com",
    "linkedin_url": "https://linkedin.com/in/johndoe",
    "github_url": "https://github.com/johndoe",
    "country": "USA",
    "city": "New York",
    "postal_code": "10001",
    "company": "Tech Corp",
    "job_title": "Senior Developer",
    "experience_level": "advanced",
    "years_experience": 5,
    "industry": "Technology",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```

### Get User by ID
```http
GET /api/v1/users/{user_id}
```
🔒 **Requires Authentication** (Own profile or Admin)

**Response:** Same as Get Current User Profile

### Update User Profile
```http
PUT /api/v1/users/profile
```
🔒 **Requires Authentication**

**Request Body:**
```json
{
  "first_name": "John",
  "last_name": "Smith",
  "bio": "Updated bio",
  "profession": "Senior Developer",
  "phone": "+1234567890",
  "website": "https://johnsmith.com",
  "linkedin_url": "https://linkedin.com/in/johnsmith",
  "github_url": "https://github.com/johnsmith",
  "country": "USA",
  "city": "San Francisco",
  "postal_code": "94105",
  "company": "New Tech Corp",
  "job_title": "Lead Developer",
  "experience_level": "expert",
  "years_experience": 7,
  "industry": "Technology"
}
```

**Response:**
```json
{
  "user_id": 1,
  "first_name": "John",
  "last_name": "Smith",
  "bio": "Updated bio",
  "profession": "Senior Developer",
  "photo_url": null,
  "objectives": null,
  "phone": "+1234567890",
  "website": "https://johnsmith.com",
  "linkedin_url": "https://linkedin.com/in/johnsmith",
  "github_url": "https://github.com/johnsmith",
  "country": "USA",
  "city": "San Francisco",
  "postal_code": "94105",
  "company": "New Tech Corp",
  "job_title": "Lead Developer",
  "experience_level": "expert",
  "years_experience": 7,
  "industry": "Technology",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

### Suspend User (Admin Only)
```http
POST /api/v1/users/{user_id}/suspend
```
🔒 **Requires Admin Authentication**

**Request Body:**
```json
{
  "reason": "Violation of terms of service"
}
```

**Response:**
```json
{
  "message": "User suspended successfully",
  "user_id": 1
}
```

### Activate User (Admin Only)
```http
POST /api/v1/users/{user_id}/activate
```
🔒 **Requires Admin Authentication**

**Response:**
```json
{
  "message": "User activated successfully",
  "user_id": 1
}
```

### Request Account Deletion
```http
DELETE /api/v1/users/me
```
🔒 **Requires Authentication**

**Response:**
```json
{
  "message": "Account deletion requested. You will receive a confirmation email."
}
```

---

## 🏥 System Routes

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development"
}
```

### Root
```http
GET /
```

**Response:**
```json
{
  "message": "E-Learning Platform API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

## 📝 Notes

- **Authentication:** Use `Authorization: Bearer <access_token>` header for protected routes
- **Token Expiry:** Access tokens expire in 30 minutes, refresh tokens in 30 days
- **Rate Limiting:** Not implemented in Phase 1 (planned for Phase 2)
- **API Documentation:** Visit `/docs` for interactive Swagger documentation
- **Error Responses:** All endpoints return appropriate HTTP status codes with error details
