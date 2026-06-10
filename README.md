# Intracranial_Aneurysm_Detection

Дипломна робота на здобуття ступеня бакалавра з комп'ютерних наук.  
Освітньо-професійна програма «Системи і методи штучного інтелекту», спеціальність 122.  
КПІ ім. Ігоря Сікорського, 2026.

## Про роботу

Розроблено інтелектуальну систему автоматизованої діагностики внутрішньочерепних аневризм за даними мультимодальних медичних зображень (CTA та MRA) на основі мереж глибокого навчання.

Система реалізує чотириетапний пайплайн від вхідних DICOM серій до структурованого діагностичного звіту:
1. Препроцесинг — ресемплінг до ізотропного простору 1×1×1 мм, модально-залежна нормалізація
2. Coarse-to-Fine сегментація Вілізієвого кола — ансамбль BasicUNet з постпроцесингом DBSCAN
3. Детекція та локалізація аневризм — бінарна детекція + класифікація по 13 анатомічних зонах
4. Генерація структурованого діагностичного звіту відповідно до вимог ESR SR

## Результати

| Модель | AUC-ROC | Macro AUC (13 зон) |
|---|---|---|
| ResNet3D-50 + MedicalNet | 0.648 | 0.611 |
| nnU-Net + CoW encoder | 0.671 | 0.624 |
| **Swin UNETR + MONAI SSL** | **0.703** | **0.641** |

Сегментатор Вілізієвого кола (Fine ансамбль): Dice 0.643, Coverage 0.918.

## Структура репозиторію

```
├── data/
│   └── preprocessing.py        
├── segmentation/
│   ├── losses.py                
│   └── dataset.py               
├── detection/
│   ├── dataset.py               
│   ├── evaluate.py              
│   └── models/
│       ├── resnet3d.py          
│       ├── nnunet.py            
│       └── swin_unetr.py        
├── report/
│   └── generate_report.py      
└── requirements.txt
```

## Датасет

[RSNA Intracranial Aneurysm Detection 2025](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) — 4348 серій CTA та MRA з експертною розміткою по 13 анатомічних зонах Вілізієвого кола.

## Встановлення

```bash
pip install -r requirements.txt
```

## Чекпоінти моделей

Чекпоінти CoW сегментатора доступні на Kaggle:  
[mariiamelika/cow-seg-checkpoint-2](https://www.kaggle.com/datasets/mariiamelika/cow-seg-checkpoint-2)