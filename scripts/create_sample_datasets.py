#!/usr/bin/env python3
"""
Create Sample Arabic Datasets
Creates sample Arabic datasets for testing different training methods when internet access is limited
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleArabicDatasetCreator:
    """Creates sample Arabic datasets for testing"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def create_sft_samples(self) -> List[Dict]:
        """Create sample SFT (Supervised Fine-Tuning) data"""
        return [
            {
                "instruction": "اكتب قصة قصيرة عن الصداقة",
                "input": "",
                "output": "كان هناك صديقان يدعيان أحمد وعلي. كانا يلعبان معاً كل يوم في الحديقة. في يوم من الأيام، سقط أحمد وجرح ركبته. ساعده علي على الوقوف ونظف جرحه. من ذلك اليوم، أصبحت صداقتهما أقوى من أي وقت مضى. تعلما أن الصداقة الحقيقية تظهر في أوقات الحاجة."
            },
            {
                "instruction": "ما هي عاصمة المملكة العربية السعودية؟",
                "input": "",
                "output": "عاصمة المملكة العربية السعودية هي الرياض. تقع الرياض في وسط المملكة وهي أكبر مدنها من حيث المساحة والسكان."
            },
            {
                "instruction": "اشرح مفهوم الذكاء الاصطناعي",
                "input": "",
                "output": "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً، مثل التعلم والاستنتاج وحل المشكلات. يشمل الذكاء الاصطناعي تقنيات مثل التعلم الآلي والشبكات العصبية ومعالجة اللغات الطبيعية."
            },
            {
                "instruction": "اكتب وصفة لطبق عربي تقليدي",
                "input": "الكبسة",
                "output": "الكبسة هي طبق أرز عربي تقليدي. المكونات: أرز بسمتي، لحم أو دجاج، بصل، طماطم، بهارات الكبسة، ملح، زيت. الطريقة: اقلي اللحم حتى ينضج، أضف البصل والطماطم، ثم الأرز والماء والبهارات. اتركه ينضج على نار هادئة لمدة 30 دقيقة."
            },
            {
                "instruction": "ما أهمية التعليم في المجتمع؟",
                "input": "",
                "output": "التعليم هو أساس تقدم المجتمعات وتطورها. يساهم في تنمية قدرات الأفراد ومهاراتهم، ويعزز الوعي والثقافة، ويساعد في حل المشكلات الاجتماعية والاقتصادية. كما يلعب دوراً مهماً في بناء شخصية الإنسان وتطوير قيمه الأخلاقية."
            }
        ]
    
    def create_dpo_samples(self) -> List[Dict]:
        """Create sample DPO (Direct Preference Optimization) data"""
        return [
            {
                "prompt": "اكتب نصيحة للطلاب قبل الامتحانات",
                "chosen": "أنصح الطلاب بالتحضير المبكر للامتحانات من خلال وضع جدول زمني للمراجعة، والحصول على قسط كافٍ من النوم، وتناول وجبات صحية، وممارسة تقنيات الاسترخاء لتقليل التوتر. كما يُنصح بحل الامتحانات السابقة والتركيز على النقاط الصعبة.",
                "rejected": "ادرس بجد واحفظ كل شيء. لا تنم كثيراً واشرب الكثير من القهوة."
            },
            {
                "prompt": "كيف يمكن المحافظة على البيئة؟",
                "chosen": "يمكن المحافظة على البيئة من خلال عدة طرق: تقليل استهلاك الطاقة، إعادة التدوير، استخدام وسائل النقل العام، زراعة الأشجار، تقليل استخدام البلاستيك، والتوعية البيئية. كل فرد يمكنه المساهمة في حماية كوكبنا للأجيال القادمة.",
                "rejected": "ازرع شجرة واحدة وستحل كل مشاكل البيئة."
            },
            {
                "prompt": "ما فوائد القراءة؟",
                "chosen": "القراءة لها فوائد عديدة: تنمي المعرفة والثقافة، تحسن مهارات التفكير النقدي، تزيد المفردات، تقلل التوتر، تحفز الخيال والإبداع، وتساعد في تطوير مهارات التواصل. كما أنها وسيلة ممتعة لقضاء الوقت واكتشاف عوالم جديدة.",
                "rejected": "القراءة مفيدة فقط للحصول على درجات جيدة في المدرسة."
            }
        ]
    
    def create_kto_samples(self) -> List[Dict]:
        """Create sample KTO (Kahneman-Tversky Optimization) data"""
        return [
            {
                "prompt": "اكتب تعليقاً على أهمية الرياضة",
                "completion": "الرياضة مهمة جداً للصحة الجسدية والنفسية. تساعد في تقوية العضلات، تحسين الدورة الدموية، وتقليل التوتر.",
                "label": True
            },
            {
                "prompt": "اكتب تعليقاً على أهمية الرياضة",
                "completion": "الرياضة مضيعة للوقت ولا فائدة منها.",
                "label": False
            },
            {
                "prompt": "صف يوماً جميلاً",
                "completion": "يوم جميل هو يوم مشمس تقضيه مع الأصدقاء والعائلة، تستمتع فيه بالطبيعة وتشعر بالسعادة والامتنان.",
                "label": True
            },
            {
                "prompt": "صف يوماً جميلاً",
                "completion": "لا يوجد أيام جميلة، كل الأيام سيئة.",
                "label": False
            }
        ]
    
    def create_evaluation_samples(self) -> List[Dict]:
        """Create sample evaluation data (multiple choice)"""
        return [
            {
                "question": "ما هي عاصمة مصر؟",
                "options": ["القاهرة", "الإسكندرية", "الجيزة", "أسوان"],
                "answer": 0,
                "explanation": "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها."
            },
            {
                "question": "كم عدد أيام السنة الميلادية؟",
                "options": ["364", "365", "366", "367"],
                "answer": 1,
                "explanation": "السنة الميلادية العادية تحتوي على 365 يوماً، والسنة الكبيسة على 366 يوماً."
            },
            {
                "question": "من كتب رواية 'مدن الملح'؟",
                "options": ["نجيب محفوظ", "عبد الرحمن منيف", "غسان كنفاني", "إميل حبيبي"],
                "answer": 1,
                "explanation": "عبد الرحمن منيف هو كاتب رواية 'مدن الملح' الشهيرة."
            }
        ]
    
    def save_dataset(self, data: List[Dict], method: str, dataset_name: str):
        """Save dataset to JSON file"""
        method_dir = self.data_dir / method
        method_dir.mkdir(exist_ok=True)
        
        file_path = method_dir / f"{dataset_name}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} samples to {file_path}")
    
    def create_all_datasets(self):
        """Create all sample datasets"""
        logger.info("Creating sample Arabic datasets...")
        
        # Create SFT datasets
        sft_data = self.create_sft_samples()
        self.save_dataset(sft_data, "sft", "arabic_sft_samples")
        
        # Create DPO datasets
        dpo_data = self.create_dpo_samples()
        self.save_dataset(dpo_data, "dpo", "arabic_dpo_samples")
        
        # Create KTO datasets
        kto_data = self.create_kto_samples()
        self.save_dataset(kto_data, "kto", "arabic_kto_samples")
        
        # Create evaluation datasets
        eval_data = self.create_evaluation_samples()
        self.save_dataset(eval_data, "evaluation", "arabic_eval_samples")
        
        # Create dataset info
        info = {
            "sft": {
                "arabic_sft_samples": {
                    "num_samples": len(sft_data),
                    "format": "instruction_response",
                    "description": "Sample Arabic instruction-response pairs for supervised fine-tuning"
                }
            },
            "dpo": {
                "arabic_dpo_samples": {
                    "num_samples": len(dpo_data),
                    "format": "preference_pairs",
                    "description": "Sample Arabic preference pairs for Direct Preference Optimization"
                }
            },
            "kto": {
                "arabic_kto_samples": {
                    "num_samples": len(kto_data),
                    "format": "binary_feedback",
                    "description": "Sample Arabic responses with binary feedback for KTO"
                }
            },
            "evaluation": {
                "arabic_eval_samples": {
                    "num_samples": len(eval_data),
                    "format": "multiple_choice",
                    "description": "Sample Arabic multiple choice questions for evaluation"
                }
            }
        }
        
        info_path = self.data_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset info saved to {info_path}")
        
        return info

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample Arabic datasets")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    
    args = parser.parse_args()
    
    creator = SampleArabicDatasetCreator(args.data_dir)
    info = creator.create_all_datasets()
    
    print("\n=== Sample Dataset Creation Complete ===")
    for method, datasets in info.items():
        print(f"\n{method.upper()}:")
        for name, dataset_info in datasets.items():
            print(f"  - {name}: {dataset_info['num_samples']} samples ({dataset_info['format']})")
    
    print(f"\nDatasets saved to: {creator.data_dir}")

if __name__ == "__main__":
    main()