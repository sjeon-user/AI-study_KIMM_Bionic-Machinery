import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime

# Step 1. 이미지 불러오기 (OpenCV BGR → RGB)
img_cv = cv2.imread('1746343803246.png')
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Step 2. Resize
new_width = 300
scale = new_width / img_rgb.shape[1]
new_height = int(img_rgb.shape[0] * scale)
img_resized = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

# Step 3. 사용자 지정 변수
target_rgb = np.array([65, 181, 162])  # 기준 색
tolerance = 50  # ± 범위 설정 변수

# Step 4. mask 조건 정의 (변수 기반 ± 범위)
lower_rgb = np.clip(target_rgb - tolerance, 0, 255)
upper_rgb = np.clip(target_rgb + tolerance, 0, 255)

print("RGB range settings:")
print("Lower:", lower_rgb)
print("Upper:", upper_rgb)

# Step 5. mask 생성
mask = cv2.inRange(img_resized, lower_rgb, upper_rgb)
num_pixels = np.count_nonzero(mask)
print("Number of pixels matching condition:", num_pixels)

# Step 6. mask 시각화
plt.imshow(mask, cmap='gray')
plt.title('Mask check')
plt.show()

# Step 7. 색 변경 (조건 만족 픽셀 모두 하나의 랜덤 색으로 변경, 나머지는 유지)
img_modified = img_resized.copy()

if num_pixels > 0:
    # 하나의 랜덤 색 생성
    single_random_color = np.random.randint(0,256, size=(3,), dtype=np.uint8)
    print("Applied random color:", single_random_color)

    # mask 조건 만족 영역에 동일한 색 할당
    img_modified[mask>0] = single_random_color
else:
    print("No pixels match the condition. Please check the mask condition.")

# Step 8. 결과 시각화 (matplotlib)
plt.imshow(img_modified)
plt.title("Condition pixels changed to one random color")
plt.axis('off')
plt.show()

# Step 9. 파일명에 datetime 추가 → 실행마다 다른 파일명 생성
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"output_selectedRGB_singleRandom_{timestamp}.png"

Image.fromarray(img_modified).save(filename)
print(f"Saved as: {filename}")

# Step 10. OpenCV GUI 출력 (영문 창 제목)
img_bgr = cv2.cvtColor(img_modified, cv2.COLOR_RGB2BGR)
cv2.imshow("Selected RGB Pixels Changed (OpenCV GUI)", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
