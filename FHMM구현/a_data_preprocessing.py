"""
1. 데이터 전처리 (Data Preprocessing)
CSV 파일에서 필요한 부분(P열, 원하는 시간대)을 자르고 저장합니다.
"""

import pandas as pd
import os


class DataPreprocessor:
    """CSV 파일에서 필요한 부분을 자르고 처리하는 클래스"""
    
    def __init__(self, data_dir='./data', output_dir='./modified data'):
        """
        Args:
            data_dir: 원본 데이터 폴더 경로
            output_dir: 처리된 데이터 저장 폴더 경로
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.names = ['CDE', 'CWE', 'DWE', 'WOE', 'HPE']
        
        # 출력 폴더 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def preprocess_all(self, num_samples=262080):
        """
        모든 가전 데이터를 전처리하고 저장
        
        Args:
            num_samples: 저장할 샘플 개수 (기본: 반년치=262080)
        """
        print("="*70)
        print("1단계: 데이터 전처리")
        print("="*70)
        
        for name in self.names:
            # 파일 경로 설정
            input_path = os.path.join(self.data_dir, f'Electricity_{name}.csv')
            output_path = os.path.join(self.output_dir, f'modified {name}.csv')
            
            if not os.path.exists(input_path):
                print(f"⚠️  파일 없음: {input_path}")
                continue
            
            # CSV 읽기
            df = pd.read_csv(input_path)
            
            # P열만 추출 (유효전력 Active Power)
            df = df[['P']]
            
            # 원하는 시간대만 자르기
            df = df.head(num_samples)
            
            # 저장
            df.to_csv(output_path, index=False)
            print(f"✓ {name}: {len(df):,}개 샘플 저장 -> {output_path}")
        
        print("\n✓ 데이터 전처리 완료!\n")
    
    def aggregate_data(self):
        """
        모든 가전의 전력을 합산하여 Total_Sumed.csv 생성
        """
        print("="*70)
        print("모든 가전 전력 합산")
        print("="*70)
        
        total_data = 0
        
        for name in self.names:
            path = os.path.join(self.output_dir, f'modified {name}.csv')
            
            if not os.path.exists(path):
                print(f"⚠️  파일 없음: {path}")
                continue
            
            df = pd.read_csv(path)
            total_data += df['P']
            print(f"✓ {name} 로드 및 합산")
        
        # 결과 저장
        result_df = pd.DataFrame({'Total_P': total_data})
        output_path = os.path.join(self.output_dir, 'Total_Sumed.csv')
        result_df.to_csv(output_path, index=False)
        print(f"\n✓ 전체 합산 데이터 저장 -> {output_path}")
        print(f"   컬럼: {result_df.columns.tolist()}")
        print(f"   샘플: {len(result_df):,}개\n")
        
        return result_df


def main():
    """메인 실행 함수"""
    preprocessor = DataPreprocessor(
        data_dir='./data',
        output_dir='./modified data'
    )
    
    # 모든 가전 전처리
    preprocessor.preprocess_all(num_samples=262080)
    
    # 전력 합산
    result_df = preprocessor.aggregate_data()
    
    return preprocessor, result_df


if __name__ == "__main__":
    preprocessor, result_df = main()
    
    # 결과 확인
    print("\n[결과 확인]")
    print(result_df.head(10))
    print(f"\n통계:")
    print(result_df.describe())
