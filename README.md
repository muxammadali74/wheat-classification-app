# 🌾 Wheat Classification App

AI asosida bug'doy kasalliklarini aniqlash ilovasi (Wheat Disease Classification). Ushbu dastur foydalanuvchilarga bug'doy bargidagi kasallikni rasm orqali aniqlash imkonini beradi.

---

## 🧠 Model haqida

- Model: `ResNet-50`
- Kirish: Rasm (jpg/png)
- Chiqish: Sog‘lom / Kasallik turlari (masalan: Rust, Scab, Healthy)
- Fayl: `model.pth`

---

## ⚙️ O‘rnatish

```bash
git clone https://github.com/foydalanuvchi_nomi/wheat-classification-app.git
cd wheat-classification-app
pip install -r requirements.txt
```

---

## 🚀 Ishga tushirish

```bash
python app.py
```

Yoki Jupyter Notebook orqali: `notebooks/test_model.ipynb`

---

## 🖼️ Foydalanish

- Ilovani ishga tushiring
- Rasm yuklang
- Kasallik turi aniqlanadi
- Natija va ishonchlilik darajasi (probability) ko‘rsatiladi

---

## 📁 Loyihaning tuzilishi

```
wheat-classification-app/
├── models
|    └── resnet50_model.pth   # Trained model
├── app.py                   # GUI yoki CLI ilova
├── requirements.txt         # Kutubxonalar
├── test_images/             # Sinov rasmlar
└── notebooks/
    └── test_model.ipynb     # Modelni o'qitish uchun notebook
```

---

## 🧪 Sinov namunasi

| Rasm                    | Tashxis | Ehtimollik |
|-------------------------|---------|------------|
| test_images/Sick.jpg    | Sick    | 98.7%      |
| test_images/healthy.jpg | Healthy | 100.0%     |

---

## 📜 Litsenziya

MIT Litsenziyasi

---

## 👤 Muallif

- **Ochilov Muxammad Ali** — Dasturchi / Talaba / Sun’iy intellekt tadqiqotchisi
- Telegram: [@ali7432](https://t.me/ali7432)

---

⭐ Agar loyiha foydali bo‘lgan bo‘lsa, yulduzcha bosing va GitHub’da repo’ni qo‘llab-quvvatlang!

