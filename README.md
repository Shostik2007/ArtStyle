# ArtStyle

ArtStyle распознаёт семь стилей живописи по изображениям. Веб-демо показывает три самых вероятных стиля, краткое описание первого и карту Grad-CAM для его предсказания.

## Локальный запуск

Положи `best_model.keras` из результатов ноутбука `03-transfer-learning.ipynb` в папку `app/`. Если веса хранятся в Git LFS, выполни `git lfs pull` перед запуском.

Команды запуска из папки проекта в PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\app\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\app\app.py
```

Открой [локальную страницу](http://localhost:8501), загрузи JPEG или PNG и включи Grad-CAM.

## Развёртывание

В Streamlit Community Cloud выбери ветку проекта с файлами приложения и укажи `app/app.py` как точку входа. Модель должна быть доступна по пути `app/best_model.keras`, а карта классов по пути `splits/label_map.json`. [Документация Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
