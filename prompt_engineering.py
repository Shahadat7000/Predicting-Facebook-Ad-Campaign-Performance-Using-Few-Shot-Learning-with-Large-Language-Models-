import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from utils import safe_to_int

logger = logging.getLogger(__name__)


class PromptEngineer:
    """
    Prompt engineering for Facebook ad campaign success prediction.
    """

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df.copy().reset_index(drop=True)

        if "is_success" not in self.train_df.columns:
            raise ValueError("Training data must contain 'is_success' column.")

        self.train_df["is_success"] = self.train_df["is_success"].apply(safe_to_int)

        self.scaler = StandardScaler()
        self.similarity_feature_cols = [
            "CTR", "CPC", "Conversion_Rate",
            "impressions", "clicks", "spent"
        ]

        self._prepare_similarity_features()

    def _safe_value(self, row: pd.Series, col: str, default=0):
        value = row.get(col, default)
        if pd.isna(value):
            return default
        return value

    def _safe_float(self, row: pd.Series, col: str, default: float = 0.0) -> float:
        value = row.get(col, default)
        if pd.isna(value):
            return default
        try:
            return float(value)
        except Exception:
            return default

    def _prepare_similarity_features(self):
        available_features = [
            col for col in self.similarity_feature_cols
            if col in self.train_df.columns
        ]

        if not available_features:
            raise ValueError("No valid similarity features found in training data.")

        self.similarity_feature_cols = available_features

        self.similarity_features = (
            self.train_df[self.similarity_feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .copy()
        )

        self.scaler.fit(self.similarity_features)

    def create_base_template(self) -> str:
        return """
You are an AI marketing analyst specializing in Facebook advertising. Your task is to evaluate whether a Facebook ad campaign was successful based on its post-campaign performance metrics.

Analyze the campaign metrics carefully and return ONLY a single digit (0 or 1).

Campaign Details:
- Age Group: {age}
- Gender: {gender}
- Interest Scores: {interest1}, {interest2}, {interest3}
- Impressions: {impressions}
- Clicks: {clicks}
- Spent: ${spent:.2f}
- CTR: {CTR:.4f}%
- CPC: ${CPC:.2f}
- Conversion Rate: {Conversion_Rate:.2f}%

Based on your marketing knowledge, return ONLY:
0 = Unsuccessful campaign
1 = Successful campaign

Your answer (0 or 1):"""

    def create_chain_of_thought_template(self) -> str:
        return """
You are an AI marketing analyst. Analyze the campaign step by step and return ONLY a single digit.

Campaign Details:
- Age Group: {age}
- Gender: {gender}
- Interests: {interest1}, {interest2}, {interest3}
- Impressions: {impressions}
- Clicks: {clicks}
- Spent: ${spent:.2f}
- CTR: {CTR:.4f}%
- CPC: ${CPC:.2f}
- Conversion Rate: {Conversion_Rate:.2f}%

Think step by step:
1. Engagement Analysis:
   - CTR of {CTR:.4f}% - is this good or poor?
   - What does this tell us about audience engagement?

2. Cost Analysis:
   - CPC of ${CPC:.2f} - is this reasonable?
   - Is the campaign cost-effective?

3. Conversion Analysis:
   - Conversion rate of {Conversion_Rate:.2f}% - is this healthy?
   - How effective is the ad at converting?

4. Overall Assessment:
   - Based on all factors, is this campaign successful?

Now, return ONLY one character:
0 = Unsuccessful
1 = Successful

Your answer (0 or 1):"""

    def select_examples_by_similarity(
        self,
        target_row: pd.Series,
        n_examples: int = 5
    ) -> pd.DataFrame:
        target_features = (
            pd.DataFrame([target_row])
            .reindex(columns=self.similarity_feature_cols)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        target_scaled = self.scaler.transform(target_features)
        train_scaled = self.scaler.transform(self.similarity_features)

        similarities = cosine_similarity(target_scaled, train_scaled)[0]

        candidate_count = min(len(self.train_df), max(n_examples * 3, n_examples))
        similar_indices = np.argsort(similarities)[-candidate_count:][::-1]

        selected_indices = []
        success_needed = n_examples // 2 + (1 if n_examples % 2 else 0)
        fail_needed = n_examples // 2

        success_count = 0
        fail_count = 0

        for idx in similar_indices:
            if idx >= len(self.train_df):
                continue

            try:
                is_success = safe_to_int(self.train_df.iloc[idx].get("is_success"))
            except Exception as e:
                logger.warning(f"Error extracting is_success at idx {idx}: {e}")
                continue

            if is_success == 1 and success_count < success_needed:
                selected_indices.append(idx)
                success_count += 1
            elif is_success == 0 and fail_count < fail_needed:
                selected_indices.append(idx)
                fail_count += 1

            if len(selected_indices) >= n_examples:
                break

        if len(selected_indices) < n_examples:
            for idx in similar_indices:
                if idx not in selected_indices and idx < len(self.train_df):
                    selected_indices.append(idx)

                if len(selected_indices) >= n_examples:
                    break

        return self.train_df.iloc[selected_indices].copy()

    def select_stratified_examples(self, n_examples: int = 5) -> pd.DataFrame:
        n_success = n_examples // 2 + (1 if n_examples % 2 else 0)
        n_fail = n_examples // 2

        successful = self.train_df[self.train_df["is_success"] == 1]
        unsuccessful = self.train_df[self.train_df["is_success"] == 0]

        selected_success = successful.sample(
            min(n_success, len(successful)),
            random_state=42
        )

        selected_fail = unsuccessful.sample(
            min(n_fail, len(unsuccessful)),
            random_state=42
        )

        examples = pd.concat([selected_success, selected_fail])

        if len(examples) < n_examples:
            additional_needed = n_examples - len(examples)
            additional = self.train_df.sample(
                additional_needed,
                replace=True,
                random_state=42
            )
            examples = pd.concat([examples, additional])

        return examples.copy()

    def format_example(self, row: pd.Series, include_reasoning: bool = False) -> str:
        success_label = "Yes" if safe_to_int(row.get("is_success")) == 1 else "No"

        ctr = self._safe_float(row, "CTR")
        cpc = self._safe_float(row, "CPC")
        conv_rate = self._safe_float(row, "Conversion_Rate")
        spent = self._safe_float(row, "spent")

        return f"""
Example:
- Age Group: {self._safe_value(row, "age", "Unknown")}
- Gender: {self._safe_value(row, "gender", "Unknown")}
- Interests: {self._safe_value(row, "interest1")}, {self._safe_value(row, "interest2")}, {self._safe_value(row, "interest3")}
- Impressions: {self._safe_value(row, "impressions")}
- Clicks: {self._safe_value(row, "clicks")}
- Spent: ${spent:.2f}
- CTR: {ctr:.4f}%
- CPC: ${cpc:.2f}
- Conversion Rate: {conv_rate:.2f}%
Prediction: {success_label}"""

    def create_few_shot_prompt(
        self,
        target_row: pd.Series,
        n_shots: int = 3,
        selection_method: str = "similarity",
        template_type: str = "chain_of_thought"
    ) -> Tuple[str, pd.DataFrame]:

        if selection_method == "similarity":
            examples_df = self.select_examples_by_similarity(target_row, n_shots)
        elif selection_method == "stratified":
            examples_df = self.select_stratified_examples(n_shots)
        else:
            examples_df = self.train_df.sample(
                min(n_shots, len(self.train_df)),
                random_state=42
            )

        examples_text = "\n\nHere are some example campaigns for reference:\n"

        for idx, (_, example_row) in enumerate(examples_df.iterrows(), 1):
            examples_text += f"\nEXAMPLE {idx}:"
            examples_text += self.format_example(example_row)

        spent = self._safe_float(target_row, "spent")
        ctr = self._safe_float(target_row, "CTR")
        cpc = self._safe_float(target_row, "CPC")
        conv_rate = self._safe_float(target_row, "Conversion_Rate")

        target_text = "\n\nNow analyze this new campaign:\n"
        target_text += "\nTARGET CAMPAIGN:"
        target_text += f"\n- Age Group: {self._safe_value(target_row, 'age', 'Unknown')}"
        target_text += f"\n- Gender: {self._safe_value(target_row, 'gender', 'Unknown')}"
        target_text += f"\n- Interests: {self._safe_value(target_row, 'interest1')}, {self._safe_value(target_row, 'interest2')}, {self._safe_value(target_row, 'interest3')}"
        target_text += f"\n- Impressions: {self._safe_value(target_row, 'impressions')}"
        target_text += f"\n- Clicks: {self._safe_value(target_row, 'clicks')}"
        target_text += f"\n- Spent: ${spent:.2f}"
        target_text += f"\n- CTR: {ctr:.4f}%"
        target_text += f"\n- CPC: ${cpc:.2f}"
        target_text += f"\n- Conversion Rate: {conv_rate:.2f}%"
        target_text += "\n\nQuestion: Is this campaign successful?"
        target_text += "\nReturn ONLY one character:"
        target_text += "\n0 = Unsuccessful"
        target_text += "\n1 = Successful"
        target_text += "\n\nYour answer (0 or 1):"

        return examples_text + target_text, examples_df

    def create_zero_shot_prompt(
        self,
        target_row: pd.Series,
        template_type: str = "chain_of_thought"
    ) -> str:
        if template_type == "chain_of_thought":
            template = self.create_chain_of_thought_template()
        else:
            template = self.create_base_template()

        return template.format(
            age=self._safe_value(target_row, "age", "Unknown"),
            gender=self._safe_value(target_row, "gender", "Unknown"),
            interest1=self._safe_value(target_row, "interest1"),
            interest2=self._safe_value(target_row, "interest2"),
            interest3=self._safe_value(target_row, "interest3"),
            impressions=self._safe_value(target_row, "impressions"),
            clicks=self._safe_value(target_row, "clicks"),
            spent=self._safe_float(target_row, "spent"),
            CTR=self._safe_float(target_row, "CTR"),
            CPC=self._safe_float(target_row, "CPC"),
            Conversion_Rate=self._safe_float(target_row, "Conversion_Rate"),
        )
