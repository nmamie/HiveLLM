import glob
import pandas as pd
from typing import Union, List, Literal
import numpy as np

from experiments.evaluator.datasets.base_dataset import BaseDataset, SwarmInput

from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from typing import Union, Literal
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class MMLUProDataset(BaseDataset):
    def __init__(
        self,
        split: Union[Literal['train'], Literal['val'], Literal['test']],
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the dataset class with the specified split and proportion sizes.

        :param split: Which split to load ('train', 'val', 'test').
        :param pro_split: Which dataset split to use from MMLU-Pro ('validation', 'test').
        :param test_size: Proportion of data to reserve for testing.
        :param val_size: Proportion of data to reserve for validation.
        :param random_state: Random state for reproducibility.
        """
        self._split = split
        
        # Load dataset
        val_split = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
        test_split = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        
        val_split = val_split.to_pandas()
        test_split = test_split.to_pandas()
        print(f"Number of questions in {val_split}: {len(val_split)}")
        print(f"Number of questions in {test_split}: {len(test_split)}")

        # Combine all splits
        self._total_df = pd.concat([val_split, test_split])
        
        # drop columns where there are not exactly 10 options
        self._total_df = self._total_df[self._total_df['options'].apply(lambda x: len(x) == 10)]

        # Split the data
        train_df, temp_df = train_test_split(
            self._total_df, test_size=test_size + val_size, random_state=random_state
        )  # Train and temp (val + test)
        val_df, test_df = train_test_split(
            temp_df, test_size=test_size / (test_size + val_size), random_state=random_state
        )  # Val and test

        # Assign the appropriate split to the dataset
        if self._split == 'train':
            self._dataset = train_df
        elif self._split == 'val':
            self._dataset = val_df
        elif self._split == 'test':
            self._dataset = test_df
        else:
            raise ValueError(f"Invalid split: {self._split}. Choose from 'train', 'val', or 'test'.")

        print(f"Number of questions in {self._split} split: {len(self._dataset)}")
        

    @staticmethod
    def get_domain() -> str:
        return 'mmlu_pro'

    # @staticmethod
    # def _load_data(
    #     data_path: str,
    #     ) -> Dataset:

    #     rng = np.random.default_rng(888)

    #     csv_paths = glob.glob(data_path + "*.csv")
    #     csv_paths = sorted(csv_paths)
    #     print("Number of topics: ", len(csv_paths))

    #     names = ['question', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'correct_answer']

    #     total_df = pd.DataFrame(columns=names)
    #     for path in csv_paths:
    #         single_df = pd.read_csv(path, header=None,
    #                         names=names)
    #         total_df = pd.concat([total_df, single_df])

    #     total_df = total_df.reset_index(drop=True)

    #     # Pseudorandom shuffle
    #     total_df = total_df.reindex(rng.permutation(total_df.index))

    #     print("Total number of questions: ", len(total_df))
        
    #     # total_df = Dataset.from_pandas(total_df)

    #     return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        # if dataset is HuggingFace Dataset
        if isinstance(self._total_df, Dataset):
            record = self._total_df[index]
            assert isinstance(record, dict)
        else:
            record = self._total_df.iloc[index]
            assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record                       

    @staticmethod
    def record_to_swarm_input(record: pd.DataFrame) -> SwarmInput:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['options'][0]} \n"
            f"Option B: {record['options'][1]} \n"
            f"Option C: {record['options'][2]} \n"
            f"Option D: {record['options'][3]} \n"
            f"Option E: {record['options'][4]} \n"
            f"Option F: {record['options'][5]} \n"
            f"Option G: {record['options'][6]} \n"
            f"Option H: {record['options'][7]} \n"
            f"Option I: {record['options'][8]} \n"
            f"Option J: {record['options'][9]} \n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer