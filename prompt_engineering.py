
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class PromptEngineer:
    """
    Comprehensive prompt engineering for Facebook ad prediction
    """

    def __init__(self, train_df: pd.DataFrame):
        """
        Initialize prompt engineer with training data
        """
        self.train_df = train_df.copy()
        self.train_df = self.train_df.reset_index(drop=True)
        self.scaler = StandardScaler()
        self._prepare_similarity_features()

        # Note: We don't pass median values to the model anymore
        # They are only used for ground truth labeling

    def _prepare_similarity_features(self):
        """Prepare features for similarity-based example selection"""
        features = ['CTR', 'CPC', 'Conversion_Rate', 'impressions', 'clicks', 'spent']
        self.similarity_features = self.train_df[features].fillna(0).copy()
        self.scaler.fit(self.similarity_features)

    def create_base_template(self) -> str:
        """
        Create base prompt template - WITHOUT mathematical formulas
        The model must learn from examples, not from explicit rules
        """
        template = """
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
        return template

    def create_chain_of_thought_template(self) -> str:
        """
        Create Chain-of-Thought prompt with strict output format (0/1)
        """
        template = """
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
        return template

    def select_examples_by_similarity(self, target_row: pd.Series, n_examples: int = 5) -> pd.DataFrame:
        """Select most similar examples from training data"""
        features = ['CTR', 'CPC', 'Conversion_Rate', 'impressions', 'clicks', 'spent']

        target_features = target_row[features].fillna(0).values.reshape(1, -1)
        target_scaled = self.scaler.transform(target_features)
        train_scaled = self.scaler.transform(self.similarity_features)

        similarities = cosine_similarity(target_scaled, train_scaled)[0]
        similar_indices = np.argsort(similarities)[-n_examples*3:][::-1]

        selected_indices = []
        success_needed = n_examples // 2 + 1
        fail_needed = n_examples // 2

        success_count = 0
        fail_count = 0

        for idx in similar_indices:
            if idx >= len(self.train_df):
                continue

            # Cast to bool explicitly in case stored as string 'True'/'False'
            is_success = bool(self.train_df.iloc[idx]['is_success'])
            if hasattr(is_success, '__class__') and isinstance(self.train_df.iloc[idx]['is_success'], str):
                is_success = self.train_df.iloc[idx]['is_success'].strip().lower() in ('true', '1', 'yes')

            if is_success and success_count < success_needed:
                selected_indices.append(idx)
                success_count += 1
            elif not is_success and fail_count < fail_needed:
                selected_indices.append(idx)
                fail_count += 1

            if len(selected_indices) >= n_examples:
                break

        # If we don't have enough, add more
        if len(selected_indices) < n_examples:
            for idx in similar_indices:
                if idx not in selected_indices and idx < len(self.train_df):
                    selected_indices.append(idx)
                    if len(selected_indices) >= n_examples:
                        break

        return self.train_df.iloc[selected_indices].copy()

    def select_stratified_examples(self, n_examples: int = 5) -> pd.DataFrame:
        """Select stratified examples (mix of successful and unsuccessful)"""
        n_success = n_examples // 2 + (1 if n_examples % 2 else 0)
        n_fail = n_examples // 2

        successful = self.train_df[self.train_df['is_success'] == True]
        unsuccessful = self.train_df[self.train_df['is_success'] == False]

        selected_success = successful.sample(min(n_success, len(successful)), random_state=42)
        selected_fail = unsuccessful.sample(min(n_fail, len(unsuccessful)), random_state=42)

        examples = pd.concat([selected_success, selected_fail])

        if len(examples) < n_examples:
            additional_needed = n_examples - len(examples)
            additional = self.train_df.sample(additional_needed, replace=True, random_state=42)
            examples = pd.concat([examples, additional])

        return examples

    def format_example(self, row: pd.Series, include_reasoning: bool = False) -> str:
        """
        Format a single example for inclusion in prompt
        Note: Examples show only the prediction (Yes/No), not reasoning
        """
        success_label = "Yes" if row['is_success'] else "No"

        ctr = row['CTR'] if pd.notna(row['CTR']) else 0
        cpc = row['CPC'] if pd.notna(row['CPC']) else 0
        conv_rate = row['Conversion_Rate'] if pd.notna(row['Conversion_Rate']) else 0

        example = f"""
Example:
- Age Group: {row['age']}
- Gender: {row['gender']}
- Interests: {row['interest1']}, {row['interest2']}, {row['interest3']}
- Impressions: {row['impressions']}
- Clicks: {row['clicks']}
- Spent: ${row['spent']:.2f}
- CTR: {ctr:.4f}%
- CPC: ${cpc:.2f}
- Conversion Rate: {conv_rate:.2f}%
Prediction: {success_label}"""

        return example

    def create_few_shot_prompt(self,
                               target_row: pd.Series,
                               n_shots: int = 3,
                               selection_method: str = 'similarity',
                               template_type: str = 'chain_of_thought') -> Tuple[str, pd.DataFrame]:
        """
        Create few-shot prompt for target campaign
        """
        # Select examples
        if selection_method == 'similarity':
            examples_df = self.select_examples_by_similarity(target_row, n_shots)
        elif selection_method == 'stratified':
            examples_df = self.select_stratified_examples(n_shots)
        else:
            examples_df = self.train_df.sample(min(n_shots, len(self.train_df)), random_state=42)

        # Get template
        if template_type == 'chain_of_thought':
            template = self.create_chain_of_thought_template()
        else:
            template = self.create_base_template()

        # Build examples section
        examples_text = "\n\nHere are some example campaigns for reference:\n"
        for idx, (_, example_row) in enumerate(examples_df.iterrows(), 1):
            examples_text += f"\nEXAMPLE {idx}:"
            examples_text += self.format_example(example_row, include_reasoning=False)

        # Format target
        target_text = f"\n\nNow analyze this new campaign:\n"
        target_text += f"\nTARGET CAMPAIGN:"
        target_text += f"\n- Age Group: {target_row['age']}"
        target_text += f"\n- Gender: {target_row['gender']}"
        target_text += f"\n- Interests: {target_row['interest1']}, {target_row['interest2']}, {target_row['interest3']}"
        target_text += f"\n- Impressions: {target_row['impressions']}"
        target_text += f"\n- Clicks: {target_row['clicks']}"
        target_text += f"\n- Spent: ${target_row['spent']:.2f}"
        target_text += f"\n- CTR: {target_row['CTR']:.4f}%"
        target_text += f"\n- CPC: ${target_row['CPC']:.2f}"
        target_text += f"\n- Conversion Rate: {target_row['Conversion_Rate']:.2f}%"
        target_text += f"\n\nQuestion: Is this campaign successful? (0/1)"

        return examples_text + target_text, examples_df

    def create_zero_shot_prompt(self, target_row: pd.Series, template_type: str = 'chain_of_thought') -> str:
        """Create zero-shot prompt (no examples)"""
        if template_type == 'chain_of_thought':
            template = self.create_chain_of_thought_template()
        else:
            template = self.create_base_template()

        return template.format(
            age=target_row['age'],
            gender=target_row['gender'],
            interest1=target_row['interest1'],
            interest2=target_row['interest2'],
            interest3=target_row['interest3'],
            impressions=target_row['impressions'],
            clicks=target_row['clicks'],
            spent=float(target_row['spent']) if pd.notna(target_row.get('spent')) else 0.0,
            CTR=float(target_row['CTR']) if pd.notna(target_row.get('CTR')) else 0.0,
            CPC=float(target_row['CPC']) if pd.notna(target_row.get('CPC')) else 0.0,
            Conversion_Rate=float(target_row['Conversion_Rate']) if pd.notna(target_row.get('Conversion_Rate')) else 0.0
        )
