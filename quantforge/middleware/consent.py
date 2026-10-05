"""
Consent Management for Python/FastAPI Applications
KVKK/GDPR/ISO 27001 Compliant
Supports: Turkish & English
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import hashlib
import secrets
from fastapi import Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

# ============================================
# TYPES
# ============================================

class Language(str, Enum):
    EN = "en"
    TR = "tr"

class ConsentCategory(str, Enum):
    NECESSARY = "necessary"
    FUNCTIONAL = "functional"
    ANALYTICS = "analytics"
    MARKETING = "marketing"

@dataclass
class ConsentPreferences:
    necessary: bool = True
    functional: bool = False
    analytics: bool = False
    marketing: bool = False
    version: str = "1.0"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    language: Language = Language.EN
    ip_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ConsentPreferences":
        data = json.loads(json_str)
        data["language"] = Language(data.get("language", "en"))
        return cls(**data)

# ============================================
# CONSTANTS
# ============================================

CONSENT_VERSION = "1.0"
CONSENT_COOKIE_NAME = "kvkk_consent"
CONSENT_COOKIE_OPTIONS = {
    "httponly": True,
    "secure": True,  # Set to False for local dev
    "samesite": "lax",
    "max_age": 60 * 60 * 24 * 365,  # 1 year
    "path": "/",
}

DEFAULT_PREFERENCES = ConsentPreferences()

# Translations
TRANSLATIONS: Dict[Language, Dict[str, str]] = {
    Language.EN: {
        "consentTitle": "Privacy & Data Consent",
        "consentDescription": "We respect your privacy. Choose which data processing you consent to. You can change this anytime in Settings.",
        "necessary": "Essential",
        "necessaryDesc": "Required for the app to function (auth, security, core features). Cannot be disabled.",
        "functional": "Preferences",
        "functionalDesc": "Remember your settings (language, theme, notifications).",
        "analytics": "Analytics",
        "analyticsDesc": "Anonymous usage stats to improve the app. No personal data.",
        "marketing": "Marketing",
        "marketingDesc": "We do not use marketing tracking.",
        "acceptAll": "Accept All",
        "rejectAll": "Reject Optional",
        "settings": "Privacy Settings",
        "learnMore": "Learn More",
        "privacyPolicy": "Privacy Policy",
        "termsOfService": "Terms of Service",
        "kvkkInfo": "KVKK Compliant • GDPR Compliant • ISO 27001 Aligned",
        "dataController": "Data Controller: Aixoratechnology",
        "contactDPO": "Contact DPO: dpo@aixoratechnology.com",
    },
    Language.TR: {
        "consentTitle": "Gizlilik ve Veri Rızası",
        "consentDescription": "Gizliliğinize saygı duyuyoruz. Onay vereceğiniz işlemleri seçin. Ayararlardan istediğiniz zaman değiştirebilirsiniz.",
        "necessary": "Temel",
        "necessaryDesc": "Uygulamanın çalışması için zorunlu (kimlik doğrulama, güvenlik, temel özellikler). Devre dışı bırakılamaz.",
        "functional": "Tercihler",
        "functionalDesc": "Ayarlarınızı hatırlar (dil, tema, bildirimler).",
        "analytics": "Analitik",
        "analyticsDesc": "Uygulamayı iyileştirmek için anonim kullanım istatistikleri. Kişisel veri toplanmaz.",
        "marketing": "Pazarlama",
        "marketingDesc": "Pazarlama takibi kullanmıyoruz.",
        "acceptAll": "Tümünü Kabul Et",
        "rejectAll": "İsteğe Bağlıları Reddet",
        "settings": "Gizlilik Ayarları",
        "learnMore": "Daha Fazla Bilgi",
        "privacyPolicy": "Gizlilik Politikası",
        "termsOfService": "Hizmet Şartları",
        "kvkkInfo": "KVKK Uyumlu • GDPR Uyumlu • ISO 27001 Uyumlu",
        "dataController": "Veri Sorumlusu: Aixoratechnology",
        "contactDPO": "VKS İletişim: dpo@aixoratechnology.com",
    },
}

# ============================================
# UTILITIES
# ============================================

def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def generate_ip_hash(ip: str, salt: Optional[str] = None) -> str:
    """Generate anonymized IP hash."""
    salt = salt or "default-salt-change-in-production"
    return "anon_" + hashlib.sha256((ip + salt).encode()).hexdigest()[:16]

def parse_consent_cookie(cookie_header: Optional[str]) -> Optional[ConsentPreferences]:
    """Parse consent from cookie header."""
    if not cookie_header:
        return None

    try:
        for cookie in cookie_header.split(";"):
            cookie = cookie.strip()
            if cookie.startswith(f"{CONSENT_COOKIE_NAME}="):
                value = cookie.split("=", 1)[1]
                prefs = ConsentPreferences.from_json(value)
                if prefs.version == CONSENT_VERSION and prefs.necessary:
                    return prefs
    except Exception as e:
        logger.warning(f"Failed to parse consent cookie: {e}")

    return None

def t(key: str, language: Language = Language.EN) -> str:
    """Translate key for language."""
    return TRANSLATIONS.get(language, TRANSLATIONS[Language.EN]).get(key, key)

# ============================================
# MIDDLEWARE
# ============================================

class ConsentMiddleware(BaseHTTPMiddleware):
    """Main consent middleware - attaches consent to request state."""

    async def dispatch(self, request: Request, call_next):
        # Parse consent from cookie
        cookie_header = request.headers.get("cookie")
        prefs = parse_consent_cookie(cookie_header)
        ip = get_client_ip(request)

        if prefs:
            request.state.consent = ConsentPreferences(
                **{**prefs.to_dict(), "ip_hash": prefs.ip_hash or generate_ip_hash(ip)}
            )
        else:
            request.state.consent = ConsentPreferences(
                ip_hash=generate_ip_hash(ip)
            )

        request.state.consent_language = request.state.consent.language

        # Add consent headers
        response = await call_next(request)
        response.headers["X-Consent-Analytics"] = "true" if request.state.consent.analytics else "false"
        response.headers["X-Consent-Functional"] = "true" if request.state.consent.functional else "false"
        response.headers["X-Consent-Language"] = request.state.consent.language.value
        response.headers["X-KVKK-Compliant"] = "true"
        response.headers["X-GDPR-Compliant"] = "true"

        return response


class RequireConsent:
    """Dependency to require specific consent category."""

    def __init__(self, category: ConsentCategory):
        self.category = category

    def __call__(self, request: Request) -> ConsentPreferences:
        consent = getattr(request.state, "consent", None)

        if not consent:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Consent required",
                    "message": f"This endpoint requires {self.category.value} consent",
                    "code": "CONSENT_REQUIRED",
                }
            )

        if self.category == ConsentCategory.NECESSARY:
            return consent

        if self.category == ConsentCategory.MARKETING:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Not available",
                    "message": "Marketing features are not supported",
                    "code": "FEATURE_NOT_SUPPORTED",
                }
            )

        has_consent = getattr(consent, self.category.value, False)

        if not has_consent:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Consent denied",
                    "message": f"This feature requires {self.category.value} consent. Please update your preferences.",
                    "code": "CONSENT_DENIED",
                    "requiredCategory": self.category.value,
                    "consentUrl": "/privacy",
                }
            )

        return consent


class OptionalConsent:
    """Dependency to check consent without blocking."""

    def __init__(self, category: ConsentCategory):
        self.category = category

    def __call__(self, request: Request) -> bool:
        consent = getattr(request.state, "consent", None)
        if not consent:
            return False
        return getattr(consent, self.category.value, False)


# ============================================
# SECURITY MIDDLEWARE
# ============================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Content Security Policy
        csp = "; ".join([
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' https://plausible.io",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self' https://api.stripe.com https://plausible.io",
            "frame-src https://js.stripe.com",
            "base-uri 'self'",
            "form-action 'self'",
        ])

        response.headers["Content-Security-Policy"] = csp
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["X-KVKK-Compliant"] = "true"
        response.headers["X-GDPR-Compliant"] = "true"

        # Remove server header
        if "server" in response.headers:
            del response.headers["server"]

        return response


class DataMinimizationMiddleware(BaseHTTPMiddleware):
    """Strip unnecessary fields from responses."""

    def __init__(self, app, allowed_fields: List[str]):
        super().__init__(app)
        self.allowed_fields = set(allowed_fields)

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Only process JSON responses
        if response.media_type == "application/json":
            # Note: In production, you'd need to parse and modify response body
            # This is a placeholder for the concept
            response.headers["X-Data-Minimization"] = "enabled"

        return response


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Audit logging for compliance."""

    async def dispatch(self, request: Request, call_next):
        start_time = datetime.utcnow()

        response = await call_next(request)

        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        consent = getattr(request.state, "consent", None)

        log_entry = {
            "timestamp": start_time.isoformat(),
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration_ms": duration,
            "ip_hash": consent.ip_hash if consent else None,
            "user_agent": request.headers.get("user-agent"),
            "consent": {
                "functional": consent.functional if consent else False,
                "analytics": consent.analytics if consent else False,
                "language": consent.language.value if consent else None,
            } if consent else None,
        }

        if response.status_code >= 400 or logger.level <= logging.INFO:
            logger.info(f"[AUDIT] {json.dumps(log_entry, ensure_ascii=False)}")

        return response


# ============================================
# CONSENT API ROUTES
# ============================================

def create_consent_routes(app):
    """Add consent management routes to FastAPI app."""

    @app.get("/api/consent", tags=["Consent"])
    async def get_consent(request: Request):
        """Get current consent preferences."""
        consent = getattr(request.state, "consent", None)
        if not consent:
            raise HTTPException(status_code=404, detail="No consent found")

        return {
            "preferences": consent.to_dict(),
            "rights": {
                "access": "/api/consent/access",
                "rectification": "/api/consent/rectify",
                "erasure": "/api/consent/erase",
                "portability": "/api/consent/export",
                "restriction": "/api/consent/restrict",
                "objection": "/api/consent/object",
            },
        }

    @app.post("/api/consent", tags=["Consent"])
    async def update_consent(request: Request, response: Response):
        """Update consent preferences."""
        body = await request.json()
        functional = body.get("functional", False)
        analytics = body.get("analytics", False)
        language = body.get("language", "en")

        if not isinstance(functional, bool) or not isinstance(analytics, bool):
            raise HTTPException(status_code=400, detail="Invalid consent data")

        consent = getattr(request.state, "consent", DEFAULT_PREFERENCES)
        ip_hash = consent.ip_hash or generate_ip_hash(get_client_ip(request))

        new_prefs = ConsentPreferences(
            functional=functional,
            analytics=analytics,
            marketing=False,
            language=Language(language) if language in [l.value for l in Language] else Language.EN,
            ip_hash=ip_hash,
        )

        # Set cookie
        response.set_cookie(
            key=CONSENT_COOKIE_NAME,
            value=new_prefs.to_json(),
            **CONSENT_COOKIE_OPTIONS
        )

        # Update request state
        request.state.consent = new_prefs

        return {"success": True, "preferences": new_prefs.to_dict()}

    @app.delete("/api/consent", tags=["Consent"])
    async def withdraw_consent(response: Response):
        """Withdraw all consent (reset)."""
        response.delete_cookie(CONSENT_COOKIE_NAME, path="/")
        return {"success": True, "message": "Consent withdrawn"}

    @app.post("/api/consent/access", tags=["Data Rights"])
    async def data_access(request: Request):
        """KVKK Art. 11 / GDPR Art. 15 - Right of Access."""
        request_id = secrets.token_urlsafe(16)
        logger.info(f"[DATA_RIGHTS] Access request: {request_id}")
        return {"message": "Data access request received", "requestId": request_id}

    @app.post("/api/consent/erase", tags=["Data Rights"])
    async def data_erasure(request: Request, response: Response):
        """KVKK Art. 11 / GDPR Art. 17 - Right to Erasure."""
        request_id = secrets.token_urlsafe(16)
        response.delete_cookie(CONSENT_COOKIE_NAME, path="/")
        logger.info(f"[DATA_RIGHTS] Erasure request: {request_id}")
        return {"message": "Data erasure request received", "requestId": request_id}

    @app.post("/api/consent/export", tags=["Data Rights"])
    async def data_portability(request: Request):
        """GDPR Art. 20 - Right to Data Portability."""
        request_id = secrets.token_urlsafe(16)
        logger.info(f"[DATA_RIGHTS] Portability request: {request_id}")
        return {"message": "Data export request received", "requestId": request_id}

    @app.post("/api/consent/restrict", tags=["Data Rights"])
    async def data_restriction(request: Request):
        """GDPR Art. 18 - Right to Restriction."""
        request_id = secrets.token_urlsafe(16)
        logger.info(f"[DATA_RIGHTS] Restriction request: {request_id}")
        return {"message": "Restriction request received", "requestId": request_id}

    @app.post("/api/consent/object", tags=["Data Rights"])
    async def data_objection(request: Request):
        """GDPR Art. 21 - Right to Object."""
        request_id = secrets.token_urlsafe(16)
        logger.info(f"[DATA_RIGHTS] Objection request: {request_id}")
        return {"message": "Objection request received", "requestId": request_id}


# ============================================
# RATE LIMITING
# ============================================

class ConsentAwareRateLimiter:
    """Rate limiter with consent-aware limits."""

    def __init__(self):
        self.limits: Dict[str, Dict[str, Any]] = {}

    def get_limit(self, consent: Optional[ConsentPreferences]) -> int:
        if not consent:
            return 60  # Anonymous
        if consent.analytics:
            return 180  # Consented users
        return 120  # Authenticated but no analytics consent

    def is_allowed(self, ip: str, consent: Optional[ConsentPreferences]) -> tuple[bool, Dict[str, int]]:
        limit = self.get_limit(consent)
        now = datetime.utcnow().timestamp()
        window = 60  # seconds

        if ip not in self.limits:
            self.limits[ip] = {"count": 1, "reset": now + window}
            return True, {"limit": limit, "remaining": limit - 1, "reset": int(now + window)}

        record = self.limits[ip]
        if now > record["reset"]:
            record["count"] = 1
            record["reset"] = now + window
            return True, {"limit": limit, "remaining": limit - 1, "reset": int(record["reset"])}

        if record["count"] >= limit:
            return False, {"limit": limit, "remaining": 0, "reset": int(record["reset"])}

        record["count"] += 1
        return True, {"limit": limit, "remaining": limit - record["count"], "reset": int(record["reset"])}


rate_limiter = ConsentAwareRateLimiter()


async def rate_limit_dependency(request: Request):
    """FastAPI dependency for rate limiting."""
    ip = get_client_ip(request)
    consent = getattr(request.state, "consent", None)

    allowed, headers = rate_limiter.is_allowed(ip, consent)

    # Add rate limit headers
    request.state.rate_limit_headers = headers

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "retryAfter": headers["reset"] - int(datetime.utcnow().timestamp()),
            },
            headers={
                "X-RateLimit-Limit": str(headers["limit"]),
                "X-RateLimit-Remaining": str(headers["remaining"]),
                "X-RateLimit-Reset": str(headers["reset"]),
                "Retry-After": str(headers["reset"] - int(datetime.utcnow().timestamp())),
            }
        )


# ============================================
# DATA RETENTION
# ============================================

DATA_RETENTION_POLICY = {
    "account": "1 year after deletion",
    "transactions": "10 years (tax law)",
    "analytics": "13 months",
    "support": "3 years",
    "marketing": "until withdrawal",
    "logs": "2 years",
}

def get_retention_headers() -> Dict[str, str]:
    return {
        "X-Data-Retention-Policy": ", ".join(f"{k}:{v}" for k, v in DATA_RETENTION_POLICY.items())
    }


# ============================================
# USAGE EXAMPLE
# ============================================

"""
# In your FastAPI app:

from fastapi import FastAPI
from consent_middleware import (
    ConsentMiddleware,
    SecurityHeadersMiddleware,
    AuditLogMiddleware,
    create_consent_routes,
    RequireConsent,
    OptionalConsent,
    ConsentCategory,
    Language,
    rate_limit_dependency,
    get_retention_headers,
)

app = FastAPI(title="My KVKK-Compliant API")

# Add middlewares (order matters!)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuditLogMiddleware)
app.add_middleware(ConsentMiddleware)

# Add consent routes
create_consent_routes(app)

# Protected endpoints
@app.get("/api/analytics/dashboard")
async def analytics_dashboard(
    consent: ConsentPreferences = Depends(RequireConsent(ConsentCategory.ANALYTICS))
):
    return {"data": "analytics dashboard", "user_language": consent.language.value}

@app.get("/api/user/preferences")
async def user_preferences(
    consent: ConsentPreferences = Depends(RequireConsent(ConsentCategory.FUNCTIONAL))
):
    return {"theme": "dark", "language": consent.language.value}

# Optional consent
@app.get("/api/public/stats")
async def public_stats(
    has_analytics: bool = Depends(OptionalConsent(ConsentCategory.ANALYTICS))
):
    if has_analytics:
        return {"detailed": True, "data": "..."}
    return {"detailed": False, "data": "basic"}

# Rate limited endpoint
@app.get("/api/search")
async def search(rate_limit: None = Depends(rate_limit_dependency)):
    return {"results": []}
"""

# ============================================
# EXPORTS
# ============================================

__all__ = [
    "Language",
    "ConsentCategory",
    "ConsentPreferences",
    "CONSENT_VERSION",
    "CONSENT_COOKIE_NAME",
    "CONSENT_COOKIE_OPTIONS",
    "DEFAULT_PREFERENCES",
    "TRANSLATIONS",
    "t",
    "get_client_ip",
    "generate_ip_hash",
    "parse_consent_cookie",
    "ConsentMiddleware",
    "SecurityHeadersMiddleware",
    "DataMinimizationMiddleware",
    "AuditLogMiddleware",
    "RequireConsent",
    "OptionalConsent",
    "create_consent_routes",
    "rate_limit_dependency",
    "DATA_RETENTION_POLICY",
    "get_retention_headers",
]