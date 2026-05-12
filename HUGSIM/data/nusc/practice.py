for i, sample in tqdm(enumerate(samples)):
    #시간 순으로 프레임 반복
    for cam in AVAILABLE_CAMERAS:
        #한 프레임 안의 여러카메라 반복
        #available_cameras <- from nusc.load import (AVAILABLE_CAMERAS ...)로 nusc/utils.py에서 미리 정의된 상수 (리스트/튜플)
        '''
        nuscene에서는 CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT 이렇게 6개의 카메라가 존재
        이때 cam은 매 반복마다 위 문자열 중 하나가 된다 
        ex. cam="CAM_FRONT", "CAM_BACK_RIGHT" 등등
        현재 처리 중인 camera 의미
        '''
        sample_data = nusc.get('sample_data', sample['data'][cam])
        '''
        #sample은 바깥반복문에서 꺼낸 현재 프레임 정보 (딕셔너리) 이게 무슨 말? : sample은 nuscene에서 제공하는 데이터셋의 한 프레임에 해당하는 정보가 담긴 딕셔너리
        #sample['data']는 그 프레임의 센서 토큰을 담은 딕셔너리 : 이건 어느 경로에서 정의? : 
        #sample['data'][cam]는 현재 처리 중인 카메라에 해당하는 센서 토큰 문자열
        #nusc.get('sample_data', token)으로 실제 sample_data 딕셔너리를 DB에서 조회 
        그결과를 sample_data 변수에 저장함 
        sample_Data는 nuscenes sdk 가 반환한 현재 camera image의 meta정보 레코드

        #nusc.get() 
        nusc : NuScenes(...)으로 만든 객체
        get : 그 객체의 매서드 (함수)
        즉 nusc.get(table_name, token)으로 
        nuscene 내부 테이블에서 해당 토큰에 맞는 레코드를 찾아 딕셔너리로 변환
        get은 nuscene 내부적으로 DB에서 데이터를 조회하는 함수
        딕셔너리 변환은 누가? : nusc.get()가 DB에서 조회한 레코드를 딕셔너리로 변환해서 반환

        대괄호 두번 : 중첩 딕셔너리 접근
        
        1. sample['data'] : sample 딕셔너리에서 data키 값을 꺼냄 
        2. sample['data'][cam] : 방금 꺼낸 딕셔너리에서 cam 키 값을 꺼냄
        '''