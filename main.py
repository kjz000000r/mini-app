"""
NutriCoach Full Mini App Backend
Полный API со ВСЕМИ функциями из final.py
"""
import os
import re
import io
import json
import hmac
import hashlib
import urllib.parse
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
import asyncio

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from openai import OpenAI
import pytesseract
from PIL import Image
import pdfplumber

load_dotenv()

app = FastAPI(
    title="NutriCoach API",
    description="Full API for Telegram Mini App",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
PAYMENT_TOKEN = os.getenv("PAYMENT_PROVIDER_TOKEN", "")
CURRENCY = os.getenv("CURRENCY", "RUB")
TRIAL_HOURS = int(os.getenv("TRIAL_HOURS", "24"))
REF_BONUS_DAYS = int(os.getenv("REF_BONUS_DAYS", "7"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "").lower()
OCR_LANG = os.getenv("OCR_LANG", "rus+eng")

# AI Client
ai = None
PRIMARY_MODEL = "deepseek/deepseek-r1-0528:free"
FALLBACK_MODELS = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-chat-v3.1:free",
    "openai/gpt-4o-mini",
]

if OPENROUTER_KEY:
    ai = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
        default_headers={
            "HTTP-Referer": "https://nutricoach.app",
            "X-Title": "NutriCoach"
        }
    )

# Database connection
def get_db():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()

# Plans configuration
PLANS = [
    {"key": "sub_7", "title": "Подписка 7 дней", "days": 7, "price_minor": 10000},
    {"key": "sub_30", "title": "Подписка 30 дней", "days": 30, "price_minor": 35000},
    {"key": "sub_90", "title": "Подписка 90 дней", "days": 90, "price_minor": 80000},
    {"key": "sub_365", "title": "Подписка 365 дней", "days": 365, "price_minor": 250000},
]

LABS_ONEOFF = {"key": "labs_350", "title": "Разовая расшифровка анализов", "price_minor": 35000}

# ============= MODELS =============

class TelegramInitData(BaseModel):
    init_data: str

class UserProfile(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: str
    weight: float = Field(ge=20, le=300)
    height: float = Field(ge=100, le=250)
    activity: str
    goal: str
    preferences: Optional[str] = None
    restrictions: Optional[str] = None

class MealCreate(BaseModel):
    text: str
    calories: int
    proteins: float
    fats: float
    carbs: float

class MealAnalyzeRequest(BaseModel):
    text: str
    image_base64: Optional[str] = None

class LabAnalysisRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None
    pdf_base64: Optional[str] = None

class RecipeRequest(BaseModel):
    products: str

class QuestionRequest(BaseModel):
    question: str

class WeightCreate(BaseModel):
    weight: float = Field(ge=20, le=300)

class PromoCodeApply(BaseModel):
    code: str

class PromoCodeCreate(BaseModel):
    code: str
    days: int = Field(ge=0, le=365)
    labs_credits: int = Field(ge=0, le=100)
    max_uses: Optional[int] = None
    expires_at: Optional[str] = None

class ChallengeLog(BaseModel):
    challenge_type: str

# ============= AUTH =============

def verify_telegram_init_data(init_data: str) -> dict:
    """Verify Telegram WebApp init data signature"""
    try:
        parsed = dict(urllib.parse.parse_qsl(init_data))
        hash_value = parsed.pop("hash", None)
        
        if not hash_value:
            raise ValueError("No hash in init_data")
        
        data_check_string = "\n".join(
            f"{k}={v}" for k, v in sorted(parsed.items())
        )
        
        secret_key = hmac.new(
            "WebAppData".encode(),
            BOT_TOKEN.encode(),
            hashlib.sha256
        ).digest()
        
        calculated_hash = hmac.new(
            secret_key,
            data_check_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if calculated_hash != hash_value:
            raise ValueError("Invalid hash")
        
        user_data = json.loads(parsed.get("user", "{}"))
        
        return {
            "user_id": user_data.get("id"),
            "username": user_data.get("username", "").lower(),
            "first_name": user_data.get("first_name"),
            "last_name": user_data.get("last_name"),
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid init_data: {str(e)}")

async def get_current_user(x_init_data: str = Header(None)):
    if not x_init_data:
        raise HTTPException(status_code=401, detail="No init_data provided")
    return verify_telegram_init_data(x_init_data)

# ============= AI HELPERS =============

async def ai_chat(system: str, user_text: str, temperature: float = 0.5) -> str:
    """Call AI with fallback models"""
    if not ai:
        return "AI не настроен"
    
    def sync_call(model: str):
        return ai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text}
            ],
            temperature=temperature
        )
    
    last_error = None
    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        try:
            response = await asyncio.to_thread(sync_call, model)
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            continue
    
    return f"Не удалось получить ответ от AI: {str(last_error)}"

async def ocr_image_bytes(img_bytes: bytes) -> str:
    """Extract text from image"""
    try:
        def sync_ocr():
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return pytesseract.image_to_string(img, lang=OCR_LANG).strip()
        return await asyncio.to_thread(sync_ocr)
    except Exception as e:
        return f"OCR error: {str(e)}"

async def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF"""
    try:
        parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text.strip())
        return "\n\n".join(parts) if parts else ""
    except Exception:
        return ""

# ============= DATABASE HELPERS =============

def save_user(db, user_id: int, username: str):
    """Save or update user"""
    try:
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO subscriptions (user_id, username) 
            VALUES (%s, %s) 
            ON CONFLICT (user_id) DO UPDATE SET username = EXCLUDED.username
            """,
            (user_id, username)
        )
        cursor.execute(
            """
            INSERT INTO credits (user_id, labs_credits) 
            VALUES (%s, 0) 
            ON CONFLICT (user_id) DO NOTHING
            """,
            (user_id,)
        )
        cursor.execute(
            """
            INSERT INTO referrals (user_id, ref_code, invited_count) 
            VALUES (%s, %s, 0) 
            ON CONFLICT (user_id) DO NOTHING
            """,
            (user_id, secrets.token_urlsafe(6))
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise

def has_access(db, user_id: int, username: str = "") -> bool:
    """Check if user has access"""
    if username == ADMIN_USERNAME and ADMIN_USERNAME:
        return True
    
    cursor = db.cursor()
    result = cursor.execute(
        "SELECT expires_at, free_until FROM subscriptions WHERE user_id = %s",
        (user_id,)
    ).fetchone()
    
    if not result:
        return False
    
    now = datetime.now(timezone.utc)
    expires_at = result.get("expires_at")
    free_until = result.get("free_until")
    
    if expires_at and expires_at > now:
        return True
    if free_until and free_until > now:
        return True
    
    return False

def activate_sub(db, user_id: int, days: int) -> datetime:
    """Activate or extend subscription"""
    cursor = db.cursor()
    
    cursor.execute(
        """
        INSERT INTO subscriptions (user_id) 
        VALUES (%s) 
        ON CONFLICT (user_id) DO NOTHING
        """,
        (user_id,)
    )
    
    result = cursor.execute(
        "SELECT expires_at FROM subscriptions WHERE user_id = %s",
        (user_id,)
    ).fetchone()
    
    now = datetime.now(timezone.utc)
    base = now
    
    if result and result.get("expires_at"):
        existing = result["expires_at"]
        if existing > now:
            base = existing
    
    new_exp = base + timedelta(days=days)
    
    cursor.execute(
        "UPDATE subscriptions SET expires_at = %s WHERE user_id = %s",
        (new_exp, user_id)
    )
    
    db.commit()
    return new_exp

def get_labs_credits(db, user_id: int) -> int:
    """Get labs credits count"""
    cursor = db.cursor()
    result = cursor.execute(
        "SELECT labs_credits FROM credits WHERE user_id = %s",
        (user_id,)
    ).fetchone()
    return result["labs_credits"] if result else 0

def consume_labs_credit(db, user_id: int) -> bool:
    """Consume one labs credit"""
    cursor = db.cursor()
    result = cursor.execute(
        "SELECT labs_credits FROM credits WHERE user_id = %s",
        (user_id,)
    ).fetchone()
    
    if not result or result["labs_credits"] <= 0:
        return False
    
    cursor.execute(
        "UPDATE credits SET labs_credits = labs_credits - 1 WHERE user_id = %s",
        (user_id,)
    )
    db.commit()
    return True

def add_labs_credit(db, user_id: int, count: int = 1):
    """Add labs credits"""
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO credits (user_id, labs_credits) 
        VALUES (%s, %s) 
        ON CONFLICT (user_id) DO UPDATE SET labs_credits = credits.labs_credits + EXCLUDED.labs_credits
        """,
        (user_id, count)
    )
    db.commit()

# ============= API ENDPOINTS =============

@app.get("/")
async def root():
    return {
        "app": "NutriCoach API",
        "version": "2.0.0",
        "status": "running"
    }

@app.post("/api/auth/verify")
async def verify_auth(data: TelegramInitData):
    """Verify and authenticate user"""
    user = verify_telegram_init_data(data.init_data)
    return {"success": True, "user": user}

@app.get("/api/user/profile")
async def get_profile(user=Depends(get_current_user), db=Depends(get_db)):
    """Get full user profile"""
    cursor = db.cursor()
    
    # Save/update user
    save_user(db, user["user_id"], user["username"])
    
    # Get subscription
    sub = cursor.execute(
        "SELECT expires_at, free_until, used_free_lab FROM subscriptions WHERE user_id = %s",
        (user["user_id"],)
    ).fetchone()
    
    # Get credits
    credits = get_labs_credits(db, user["user_id"])
    
    # Get achievements count
    ach_count = cursor.execute(
        "SELECT COUNT(*) as count FROM achievements WHERE user_id = %s",
        (user["user_id"],)
    ).fetchone()["count"]
    
    # Get referral info
    ref_info = cursor.execute(
        "SELECT ref_code, invited_count FROM referrals WHERE user_id = %s",
        (user["user_id"],)
    ).fetchone()
    
    return {
        "user": user,
        "subscription": {
            "expires_at": sub["expires_at"].isoformat() if sub and sub.get("expires_at") else None,
            "free_until": sub["free_until"].isoformat() if sub and sub.get("free_until") else None,
            "has_access": has_access(db, user["user_id"], user["username"]),
            "used_free_lab": sub["used_free_lab"] if sub else False
        },
        "labs_credits": credits,
        "achievements_count": ach_count,
        "referral": {
            "code": ref_info["ref_code"] if ref_info else None,
            "invited_count": ref_info["invited_count"] if ref_info else 0
        }
    }

@app.post("/api/user/activate-trial")
async def activate_trial(user=Depends(get_current_user), db=Depends(get_db)):
    """Activate trial period"""
    cursor = db.cursor()
    
    # Check if already used
    result = cursor.execute(
        "SELECT free_until FROM subscriptions WHERE user_id = %s",
        (user["user_id"],)
    ).fetchone()
    
    if result and result.get("free_until"):
        raise HTTPException(status_code=400, detail="Trial already used")
    
    # Activate trial
    until = datetime.now(timezone.utc) + timedelta(hours=TRIAL_HOURS)
    cursor.execute(
        """
        INSERT INTO subscriptions (user_id, free_until) 
        VALUES (%s, %s) 
        ON CONFLICT (user_id) DO UPDATE SET free_until = EXCLUDED.free_until
        """,
        (user["user_id"], until)
    )
    db.commit()
    
    return {
        "success": True,
        "free_until": until.isoformat(),
        "hours": TRIAL_HOURS
    }

# ============= MEALS =============

@app.get("/api/meals")
async def get_meals(
    period: str = "today",
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Get meals for period"""
    cursor = db.cursor()
    
    now = datetime.now(timezone.utc)
    if period == "today":
        start = datetime.combine(now.date(), datetime.min.time()).replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)
    elif period == "week":
        start = now - timedelta(days=7)
        end = now
    else:
        start = now - timedelta(days=30)
        end = now
    
    meals = cursor.execute(
        """
        SELECT id, ts, text, calories, proteins, fats, carbs 
        FROM meals 
        WHERE user_id = %s AND ts >= %s AND ts < %s 
        ORDER BY ts DESC
        """,
        (user["user_id"], start, end)
    ).fetchall()
    
    total = {
        "calories": sum(m["calories"] or 0 for m in meals),
        "proteins": sum(m["proteins"] or 0 for m in meals),
        "fats": sum(m["fats"] or 0 for m in meals),
        "carbs": sum(m["carbs"] or 0 for m in meals),
    }
    
    return {
        "meals": [
            {
                "id": m["id"],
                "timestamp": m["ts"].isoformat(),
                "text": m["text"],
                "calories": m["calories"],
                "proteins": m["proteins"],
                "fats": m["fats"],
                "carbs": m["carbs"]
            }
            for m in meals
        ],
        "total": total
    }

@app.post("/api/meals")
async def create_meal(
    meal: MealCreate,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Create new meal"""
    if not has_access(db, user["user_id"], user["username"]):
        raise HTTPException(status_code=403, detail="No access")
    
    cursor = db.cursor()
    result = cursor.execute(
        """
        INSERT INTO meals (user_id, ts, text, calories, proteins, fats, carbs)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            user["user_id"],
            datetime.now(timezone.utc),
            meal.text,
            meal.calories,
            meal.proteins,
            meal.fats,
            meal.carbs
        )
    )
    meal_id = result.fetchone()["id"]
    db.commit()
    
    # Award achievements
    award_achievements(db, user["user_id"])
    
    return {"success": True, "meal_id": meal_id}

@app.delete("/api/meals/{meal_id}")
async def delete_meal(
    meal_id: int,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Delete meal"""
    cursor = db.cursor()
    cursor.execute(
        "DELETE FROM meals WHERE id = %s AND user_id = %s",
        (meal_id, user["user_id"])
    )
    db.commit()
    return {"success": True}

@app.post("/api/meals/analyze")
async def analyze_meal(
    request: MealAnalyzeRequest,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Analyze meal with AI"""
    if not has_access(db, user["user_id"], user["username"]):
        raise HTTPException(status_code=403, detail="No access")
    
    text = request.text
    
    # If image provided, extract text
    if request.image_base64:
        import base64
        img_data = base64.b64decode(request.image_base64.split(",")[1] if "," in request.image_base64 else request.image_base64)
        ocr_text = await ocr_image_bytes(img_data)
        text = f"{text}\n{ocr_text}".strip()
    
    prompt = f"""
Оцени прием пищи и верни ТОЛЬКО JSON в формате:
{{"calories": int, "proteins": float, "fats": float, "carbs": float, "summary": "краткое описание"}}

Текст: {text}
"""
    
    response = await ai_chat(
        "Ты нутрициолог. Верни ТОЛЬКО JSON без markdown.",
        prompt,
        0.2
    )
    
    try:
        # Clean response
        cleaned = response.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return {
            "calories": int(data.get("calories", 0)),
            "proteins": float(data.get("proteins", 0)),
            "fats": float(data.get("fats", 0)),
            "carbs": float(data.get("carbs", 0)),
            "summary": str(data.get("summary", "Прием пищи"))
        }
    except:
        return {
            "calories": 0,
            "proteins": 0.0,
            "fats": 0.0,
            "carbs": 0.0,
            "summary": "Не удалось распознать"
        }

# ============= NUTRITION PLAN =============

@app.post("/api/nutrition-plan")
async def create_nutrition_plan(
    profile: UserProfile,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Create personalized nutrition plan"""
    if not has_access(db, user["user_id"], user["username"]):
        raise HTTPException(status_code=403, detail="No access")
    
    # Calculate BMR
    if profile.sex == "male":
        bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age + 5
    else:
        bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age - 161
    
    # Activity multiplier
    activity_mult = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "high": 1.725,
        "extreme": 1.9
    }
    multiplier = activity_mult.get(profile.activity.lower(), 1.55)
    
    # Goal adjustment
    goal_adj = {
        "снижение веса": 0.85,
        "поддержание": 1.0,
        "набор": 1.15
    }
    adjustment = goal_adj.get(profile.goal.lower(), 1.0)
    
    daily_calories = round(bmr * multiplier * adjustment)
    
    prompt = f"""
Составь 7-дневный персональный план питания (завтрак/обед/ужин/перекусы).

Данные клиента:
- Возраст: {profile.age}
- Пол: {profile.sex}
- Вес: {profile.weight} кг
- Рост: {profile.height} см
- Активность: {profile.activity}
- Цель: {profile.goal}
- Предпочтения: {profile.preferences or 'нет'}
- Ограничения: {profile.restrictions or 'нет'}
- Рекомендуемая калорийность: {daily_calories} ккал/день

Пиши структурированно, с граммовками и калориями для каждого приема пищи.
"""
    
    plan_text = await ai_chat(
        "Ты профессиональный нутрициолог. Пиши структурировано.",
        prompt,
        0.4
    )
    
    return {
        "plan": plan_text,
        "daily_calories": daily_calories,
        "bmr": round(bmr)
    }

# ============= LABS ANALYSIS =============

@app.post("/api/labs/analyze")
async def analyze_labs(
    request: LabAnalysisRequest,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Analyze lab results"""
    cursor = db.cursor()
    
    # Check if admin
    is_admin = user["username"] == ADMIN_USERNAME and ADMIN_USERNAME
    
    if not is_admin:
        # Check free lab
        sub = cursor.execute(
            "SELECT used_free_lab FROM subscriptions WHERE user_id = %s",
            (user["user_id"],)
        ).fetchone()
        
        if not sub or not sub["used_free_lab"]:
            # Use free lab
            cursor.execute(
                "UPDATE subscriptions SET used_free_lab = TRUE WHERE user_id = %s",
                (user["user_id"],)
            )
            db.commit()
        else:
            # Check credits
            if not consume_labs_credit(db, user["user_id"]):
                raise HTTPException(status_code=403, detail="No credits")
    
    # Extract text
    text = request.text or ""
    
    if request.image_base64:
        import base64
        img_data = base64.b64decode(request.image_base64.split(",")[1] if "," in request.image_base64 else request.image_base64)
        ocr_text = await ocr_image_bytes(img_data)
        text = f"{text}\n{ocr_text}".strip()
    
    if request.pdf_base64:
        import base64
        pdf_data = base64.b64decode(request.pdf_base64.split(",")[1] if "," in request.pdf_base64 else request.pdf_base64)
        pdf_text = await ocr_pdf_bytes(pdf_data)
        text = f"{text}\n{pdf_text}".strip()
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text to analyze")
    
    prompt = f"""
Ты нутрициолог. Проанализируй лабораторные анализы и дай практические рекомендации.

{text}
"""
    
    analysis = await ai_chat(
        "Пиши кратко и структурированно.",
        prompt,
        0.3
    )
    
    return {"analysis": analysis}

# ============= RECIPES =============

@app.post("/api/recipes")
async def get_recipes(
    request: RecipeRequest,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Generate recipes from products"""
    if not has_access(db, user["user_id"], user["username"]):
        raise HTTPException(status_code=403, detail="No access")
    
    prompt = f"""
На основе списка продуктов составь 3 рецепта. Для каждого: название, ингредиенты с граммовками, шаги приготовления, калорийность и БЖУ на порцию.

Продукты:
{request.products}
"""
    
    recipes = await ai_chat(
        "Ты шеф-повар и нутрициолог. Пиши структурированно, ясно.",
        prompt,
        0.5
    )
    
    return {"recipes": recipes}

# ============= Q&A =============

@app.post("/api/question")
async def ask_question(
    request: QuestionRequest,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Ask nutrition question"""
    if not has_access(db, user["user_id"], user["username"]):
        raise HTTPException(status_code=403, detail="No access")
    
    prompt = f"Ты нутрициолог. Дай краткий, практический ответ на вопрос:\n\n{request.question}"
    
    answer = await ai_chat(
        "Кратко и по делу.",
        prompt,
        0.5
    )
    
    return {"answer": answer}

# ============= WEIGHT TRACKING =============

@app.post("/api/weight")
async def log_weight(
    data: WeightCreate,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Log weight"""
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO weight_tracking (user_id, weight, ts) VALUES (%s, %s, %s)",
        (user["user_id"], data.weight, datetime.now(timezone.utc))
    )
    db.commit()
    return {"success": True}

@app.get("/api/weight/history")
async def get_weight_history(
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Get weight history"""
    cursor = db.cursor()
    weights = cursor.execute(
        """
        SELECT weight, ts FROM weight_tracking
        WHERE user_id = %s
        ORDER BY ts DESC
        LIMIT 30
        """,
        (user["user_id"],)
    ).fetchall()
    
    return {
        "history": [
            {
                "weight": w["weight"],
                "date": w["ts"].isoformat()
            }
            for w in weights
        ]
    }

# ============= STATISTICS =============

@app.get("/api/stats")
async def get_stats(
    period: str = "week",
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Get statistics"""
    cursor = db.cursor()
    
    now = datetime.now(timezone.utc)
    if period == "week":
        since = now - timedelta(days=7)
    else:
        since = now - timedelta(days=30)
    
    daily_stats = cursor.execute(
        """
        SELECT 
            DATE(ts) as day,
            SUM(calories) as calories,
            SUM(proteins) as proteins,
            SUM(fats) as fats,
            SUM(carbs) as carbs
        FROM meals
        WHERE user_id = %s AND ts >= %s
        GROUP BY DATE(ts)
        ORDER BY DATE(ts)
        """,
        (user["user_id"], since)
    ).fetchall()
    
    return {
        "daily_stats": [
            {
                "date": stat["day"].isoformat(),
                "calories": stat["calories"] or 0,
                "proteins": stat["proteins"] or 0,
                "fats": stat["fats"] or 0,
                "carbs": stat["carbs"] or 0
            }
            for stat in daily_stats
        ]
    }

# ============= ACHIEVEMENTS =============

@app.get("/api/achievements")
async def get_achievements(
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Get achievements"""
    cursor = db.cursor()
    achievements = cursor.execute(
        "SELECT badge, ts FROM achievements WHERE user_id = %s ORDER BY ts DESC",
        (user["user_id"],)
    ).fetchall()
    
    return {
        "achievements": [
            {
                "badge": a["badge"],
                "earned_at": a["ts"].isoformat()
            }
            for a in achievements
        ]
    }

def award_achievements(db, user_id: int):
    """Award achievements based on activity"""
    cursor = db.cursor()
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    
    # Get meals from last week
    meals = cursor.execute(
        "SELECT ts, text FROM meals WHERE user_id = %s AND ts >= %s ORDER BY ts",
        (user_id, week_ago)
    ).fetchall()
    
    # Breakfast hero
    breakfast_days = set()
    for m in meals:
        hour = m["ts"].hour
        if 5 <= hour < 10:
            breakfast_days.add(m["ts"].date())
    
    if len(breakfast_days) >= 7:
        try:
            cursor.execute(
                """
                INSERT INTO achievements (user_id, badge, ts) 
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (user_id, "Завтрак-герой", now)
            )
        except:
            pass
    
    db.commit()

# ============= CHALLENGES =============

@app.get("/api/challenges")
async def get_challenges(
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Get user challenges"""
    cursor = db.cursor()
    challenges = cursor.execute(
        """
        SELECT challenge_type, progress, completed, start_date
        FROM challenges
        WHERE user_id = %s
        """,
        (user["user_id"],)
    ).fetchall()
    
    return {
        "challenges": [
            {
                "type": c["challenge_type"],
                "progress": c["progress"],
                "completed": c["completed"],
                "start_date": c["start_date"].isoformat() if c["start_date"] else None
            }
            for c in challenges
        ]
    }

@app.post("/api/challenges/{challenge_type}/start")
async def start_challenge(
    challenge_type: str,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Start a challenge"""
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO challenges (user_id, challenge_type, start_date, progress, completed)
        VALUES (%s, %s, %s, 0, 0)
        ON CONFLICT (user_id, challenge_type) DO UPDATE
        SET start_date = EXCLUDED.start_date, progress = 0, completed = 0
        """,
        (user["user_id"], challenge_type, datetime.now(timezone.utc))
    )
    db.commit()
    return {"success": True}

@app.post("/api/challenges/{challenge_type}/log")
async def log_challenge(
    challenge_type: str,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Log challenge progress"""
    cursor = db.cursor()
    today = datetime.now(timezone.utc).date()
    
    # Check if already logged today
    exists = cursor.execute(
        """
        SELECT 1 FROM challenge_logs
        WHERE user_id = %s AND challenge_type = %s AND log_date = %s
        """,
        (user["user_id"], challenge_type, today)
    ).fetchone()
    
    if exists:
        return {"success": False, "message": "Already logged today"}
    
    # Log
    cursor.execute(
        """
        INSERT INTO challenge_logs (user_id, challenge_type, log_date, completed)
        VALUES (%s, %s, %s, TRUE)
        """,
        (user["user_id"], challenge_type, today)
    )
    
    # Update progress
    cursor.execute(
        """
        UPDATE challenges 
        SET progress = progress + 1
        WHERE user_id = %s AND challenge_type = %s
        """,
        (user["user_id"], challenge_type)
    )
    
    # Check completion
    progress = cursor.execute(
        "SELECT progress FROM challenges WHERE user_id = %s AND challenge_type = %s",
        (user["user_id"], challenge_type)
    ).fetchone()["progress"]
    
    if progress >= 7:
        cursor.execute(
            "UPDATE challenges SET completed = 1 WHERE user_id = %s AND challenge_type = %s",
            (user["user_id"], challenge_type)
        )
        
        # Award achievement
        cursor.execute(
            """
            INSERT INTO achievements (user_id, badge, ts)
            VALUES (%s, %s, %s)
            """,
            (user["user_id"], f"Челлендж: {challenge_type}", datetime.now(timezone.utc))
        )
    
    db.commit()
    
    return {
        "success": True,
        "progress": progress,
        "completed": progress >= 7
    }

# ============= PROMOCODES =============

@app.post("/api/promo/apply")
async def apply_promo(
    data: PromoCodeApply,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Apply promocode"""
    cursor = db.cursor()
    
    promo = cursor.execute(
        """
        SELECT days, labs_credits, max_uses, used_count, expires_at
        FROM promocodes
        WHERE code = %s
        """,
        (data.code,)
    ).fetchone()
    
    if not promo:
        raise HTTPException(status_code=404, detail="Promocode not found")
    
    # Check expiry
    if promo["expires_at"] and promo["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Promocode expired")
    
    # Check uses
    if promo["max_uses"] and promo["used_count"] >= promo["max_uses"]:
        raise HTTPException(status_code=400, detail="Promocode limit reached")
    
    days = promo["days"]
    credits = promo["labs_credits"]
    
    if days <= 0 and credits <= 0:
        raise HTTPException(status_code=400, detail="Invalid promocode")
    
    # Apply
    if days > 0:
        activate_sub(db, user["user_id"], days)
    
    if credits > 0:
        add_labs_credit(db, user["user_id"], credits)
    
    # Update usage
    cursor.execute(
        "UPDATE promocodes SET used_count = used_count + 1 WHERE code = %s",
        (data.code,)
    )
    db.commit()
    
    return {
        "success": True,
        "days": days,
        "credits": credits
    }

@app.post("/api/promo/create")
async def create_promo(
    data: PromoCodeCreate,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Create promocode (admin only)"""
    if user["username"] != ADMIN_USERNAME or not ADMIN_USERNAME:
        raise HTTPException(status_code=403, detail="Admin only")
    
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO promocodes (code, days, labs_credits, max_uses, used_count, expires_at)
        VALUES (%s, %s, %s, %s, 0, %s)
        ON CONFLICT (code) DO UPDATE SET
            days = EXCLUDED.days,
            labs_credits = EXCLUDED.labs_credits,
            max_uses = EXCLUDED.max_uses,
            expires_at = EXCLUDED.expires_at
        """,
        (data.code, data.days, data.labs_credits, data.max_uses, data.expires_at)
    )
    db.commit()
    
    return {"success": True}

# ============= PLANS & PAYMENT =============

@app.get("/api/plans")
async def get_plans():
    """Get available subscription plans"""
    return {
        "plans": PLANS,
        "labs_oneoff": LABS_ONEOFF,
        "currency": CURRENCY
    }

@app.get("/api/referral")
async def get_referral_info(
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Get referral information"""
    cursor = db.cursor()
    ref = cursor.execute(
        "SELECT ref_code, invited_count FROM referrals WHERE user_id = %s",
        (user["user_id"],)
    ).fetchone()
    
    if not ref:
        # Generate code
        code = secrets.token_urlsafe(6)
        cursor.execute(
            """
            INSERT INTO referrals (user_id, ref_code, invited_count)
            VALUES (%s, %s, 0)
            """,
            (user["user_id"], code)
        )
        db.commit()
        ref = {"ref_code": code, "invited_count": 0}
    
    # Get bot info for link
    bot_username = "nutricoach_bot"  # Replace with your bot username
    ref_link = f"t.me/{bot_username}?start={ref['ref_code']}"
    
    return {
        "ref_code": ref["ref_code"],
        "invited_count": ref["invited_count"],
        "ref_link": ref_link,
        "bonus_days": REF_BONUS_DAYS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
