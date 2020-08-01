import numpy as np
import cv2


# ===========================================
# common
def define_search_window(img_shape, block_position, window_size, block_size):
    [x, y] = block_position
    left = x + block_size / 2 - window_size / 2
    right = x + block_size / 2 + window_size / 2
    up = y + block_size / 2 - window_size / 2
    down = y + block_size / 2 + window_size / 2

    if left < 0:
        left = 0
    if right > img_shape[0]:
        left = img_shape[0] - window_size
    if up < 0:
        up = 0
    if down > img_shape[1]:
        up = img_shape[1] - window_size

    return np.array([int(left), int(up)])


def set_kaiser(block_size, beta_kaiser):
    k = np.kaiser(block_size, beta_kaiser)
    k = np.reshape(k, (1, len(k)))

    return np.matmul(np.transpose(k), k)


def locate_block(i, j, block_step, block_size, width, height):
    if i * block_step + block_size < width:
        x = i * block_step
    else:
        x = width - block_size

    if j * block_step + block_size < height:
        y = j * block_step
    else:
        y = height - block_size

    return np.array([x, y], dtype=np.int)


def aggregate_hard_threshold(similar_blocks, block_positions,
                             result, weight, nonzero_cnt, count, kaiser):
    dim = similar_blocks.shape
    if nonzero_cnt < 1:
        nonzero_cnt = 1
    block_weight = (1 / nonzero_cnt) * kaiser
    for i in range(count):
        [x, y] = block_positions[i, :]
        sim = similar_blocks[i, :, :]
        img = (1 / nonzero_cnt) * cv2.idct(sim) * kaiser
        result[x:x + dim[1], y:y + dim[2]] += img
        weight[x:x + dim[1], y:y + dim[2]] += block_weight

    return result, weight


def aggregate_wiener(similar_blocks, wiener_weight, block_positions,
                     result, weight, count):
    dim = similar_blocks.shape
    for i in range(count):
        [x, y] = block_positions[i, :]
        block = wiener_weight * cv2.idct(similar_blocks[i, :, :])
        result[x:x + dim[1], y:y + dim[2]] += block
        weight[x:x + dim[1], y:y + dim[2]] += wiener_weight

    return result, weight
# ===========================================


# ===========================================
# step 1
def first_fast_match(img, first_block_position, block_size,
                     search_step, threshold, max_matches, window_size):
    final_block_positions = np.zeros((max_matches, 2), dtype=np.int)
    final_similar_blocks = np.zeros((max_matches, block_size, block_size))

    [x, y] = first_block_position
    dct_template = cv2.dct(img[x:x + block_size, y:y + block_size])

    final_block_positions[0, :] = first_block_position
    final_similar_blocks[0, :, :] = dct_template

    window = define_search_window(img.shape, first_block_position, window_size, block_size)
    [x, y] = window
    block_count = int((window_size - block_size) / search_step)

    similar_block = np.zeros((block_count ** 2, block_size, block_size))
    block_positions = np.zeros((block_count ** 2, 2), dtype=np.int)
    distances = np.zeros(block_count ** 2)
    matched = 0

    for i in range(block_count):
        for j in range(block_count):
            dct_block = cv2.dct(img[x:x + block_size, y:y + block_size])
            dist = np.linalg.norm(dct_template - dct_block) ** 2 / (block_size ** 2)

            if 0 < dist < threshold:
                similar_block[matched, :, :] = dct_block
                block_positions[matched, :] = (x, y)
                distances[matched] = dist
                matched += 1
            y += search_step
        x += search_step
        y = window[1]

    distances = distances[:matched]
    sorted_indices = distances.argsort()

    if matched < max_matches:
        count = matched + 1
    else:
        count = max_matches

    if count > 0:
        for i in range(1, count):
            index = sorted_indices[i - 1]
            final_similar_blocks[i, :, :] = similar_block[index, :, :]
            final_block_positions[i, :] = block_positions[index, :]

    return final_similar_blocks, final_block_positions, count


def first_3d_filtering(similar_blocks, threshold):
    nonzero_count = 0
    dim = similar_blocks.shape

    for i in range(dim[1]):
        for j in range(dim[2]):
            dct = cv2.dct(similar_blocks[:, i, j])
            dct[np.abs(dct[:]) < threshold] = 0
            nonzero_count += dct.nonzero()[0].size
            similar_blocks[:, i, j] = cv2.idct(dct)[0]

    return similar_blocks, nonzero_count


def bm3d_first(img, block_size, block_step, beta_kaiser,
               search_step, threshold_bm, max_matches, window_size, threshold_3d):
    (w, h) = img.shape
    w_count = (w - block_size) / block_step
    h_count = (h - block_size) / block_step

    result = np.zeros(img.shape)
    weight = np.zeros(img.shape)
    kaiser = set_kaiser(block_size, beta_kaiser)

    for i in range(int(w_count + 1)):
        for j in range(int(h_count + 1)):
            first_block_position = locate_block(i, j, block_step, block_size,
                                                w, h)
            similar, position, count = first_fast_match(
                img,
                first_block_position,
                block_size,
                search_step,
                threshold_bm,
                max_matches,
                window_size
            )

            similar, nonzero_cnt = first_3d_filtering(similar, threshold_3d)
            result, weight = aggregate_hard_threshold(
                similar,
                position,
                result,
                weight,
                nonzero_cnt,
                count,
                kaiser
            )

    return result / weight
# ===========================================


# ===========================================
# step 2
def second_fast_match(first_result, img, first_block_position, block_size,
                      search_step, threshold, max_matches, window_size):
    final_block_positions = np.zeros((max_matches, 2), dtype=np.int)
    final_similar_blocks = np.zeros((max_matches, block_size, block_size))
    final_noisy_blocks = np.zeros((max_matches, block_size, block_size))

    [x, y] = first_block_position
    dct_template = cv2.dct(first_result[x:x + block_size, y:y + block_size])
    noisy_template = cv2.dct(img[x:x + block_size, y:y + block_size])

    final_block_positions[0, :] = first_block_position
    final_similar_blocks[0, :, :] = dct_template
    final_noisy_blocks[0, :, :] = noisy_template

    window = define_search_window(img.shape, first_block_position, window_size, block_size)
    [x, y] = window
    block_count = int((window_size - block_size) / search_step)

    similar_block = np.zeros((block_count ** 2, block_size, block_size))
    block_positions = np.zeros((block_count ** 2, 2), dtype=np.int)
    distances = np.zeros(block_count ** 2)
    matched = 0

    for i in range(block_count):
        for j in range(block_count):
            dct_block = cv2.dct(first_result[x:x + block_size, y:y + block_size])
            dist = np.linalg.norm(dct_template - dct_block) ** 2 / (block_size ** 2)

            if 0 < dist < threshold:
                similar_block[matched, :, :] = dct_block
                block_positions[matched, :] = (x, y)
                distances[matched] = dist
                matched += 1
            y += search_step
        x += search_step
        y = window[1]

    distances = distances[:matched]
    sorted_indices = distances.argsort()

    if matched < max_matches:
        count = matched + 1
    else:
        count = max_matches

    if count > 0:
        for i in range(1, count):
            index = sorted_indices[i - 1]
            [x, y] = block_positions[index, :]
            final_similar_blocks[i, :, :] = similar_block[index, :, :]
            final_noisy_blocks[i, :, :] = cv2.dct(img[x:x + block_size, y:y + block_size])
            final_block_positions[i, :] = block_positions[index, :]

    return final_similar_blocks, final_noisy_blocks, final_block_positions, count


def second_3d_filtering(similar_blocks, noisy_blocks, sigma):
    dim = similar_blocks.shape
    wiener_weight = np.zeros((dim[1], dim[2]))

    for i in range(dim[1]):
        for j in range(dim[2]):
            dct = cv2.dct(similar_blocks[:, i, j])
            norm = np.dot(np.transpose(dct), dct)
            weight = norm / (norm + sigma ** 2)

            if weight != 0:
                wiener_weight[i, j] = 1 / (weight ** 2 * sigma ** 2)

            noisy_dct = cv2.dct(noisy_blocks[:, i, j]) * weight
            similar_blocks[:, i, j] = cv2.idct(noisy_dct)[0]

    return similar_blocks, wiener_weight


def bm3d_second(first_result, img, block_size, block_step,
                search_step, threshold_bm, max_matches, window_size, sigma):
    (w, h) = img.shape
    w_count = (w - block_size) / block_step
    h_count = (h - block_size) / block_step

    result = np.zeros(img.shape)
    weight = np.zeros(img.shape)

    for i in range(int(w_count + 1)):
        for j in range(int(h_count + 1)):
            first_block_position = locate_block(i, j, block_step, block_size,
                                                w, h)
            similar, noisy, position, count = second_fast_match(
                first_result,
                img,
                first_block_position,
                block_size,
                search_step,
                threshold_bm,
                max_matches,
                window_size
            )

            similar, wiener_weight = second_3d_filtering(similar, noisy, sigma)
            result, weight = aggregate_wiener(
                similar,
                wiener_weight,
                position,
                result,
                weight,
                count
            )

    return result / weight
# ===========================================


# ===========================================
# combined
def bm3d(img, first_params, second_params, sigma):
    first_result = bm3d_first(
        img,
        first_params.get('block_size', 8),
        first_params['block_step'],
        first_params['beta_kaiser'],
        first_params['search_step'],
        first_params['threshold_bm'],
        first_params['max_matches'],
        first_params['window_size'],
        first_params['threshold_3d']
    )

    final_result = bm3d_second(
        first_result,
        img,
        second_params['block_size'],
        second_params['block_step'],
        second_params['search_step'],
        second_params['threshold_bm'],
        second_params['max_matches'],
        second_params['window_size'],
        sigma
    )

    return final_result.astype(np.uint8)
# ===========================================


if __name__ == '__main__':
    from gaussian_noise import add_gaussian

    img = cv2.imread('./test_assets/base/512_lenna.png', 0)
    img = img / 255
    noisy, noise = add_gaussian(img, sigma=15/255)

    sigma = 25
    first_params = {
        'block_size': 8,
        'block_step': 3,
        'beta_kaiser': 2.0,
        'search_step': 3,
        'threshold_bm': 3000,
        'max_matches': 16,
        'window_size': 32,
        'threshold_3d': 2.7 * sigma
    }

    second_params = {
        'block_size': 8,
        'block_step': 3,
        'search_step': 3,
        'threshold_bm': 4000,
        'max_matches': 32,
        'window_size': 32
    }

    result = bm3d(noisy, first_params, second_params, sigma)

    cv2.imshow('Test', result)
    cv2.waitKey(0)
