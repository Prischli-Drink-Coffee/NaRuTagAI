import torch
from concurrent.futures import ProcessPoolExecutor, as_completed


def create_audio_chunks_and_masks(audios, audios_mask, chunk_size):
    """
    Разбивает аудио на чанки равного размера и возвращает соответствующие маски.

    :param audios: Tensor с аудио данными, размерность [batch_size, num_channels, length]
    :param audios_mask: Tensor с масками, размерность [batch_size, length]
    :param chunk_size: Размер чанка
    :return: Чанки аудио и соответствующие маски
    """
    # Удаляем лишнюю размерность, если она есть

    batch_size, length = audios.size()

    # Вычисляем количество полных чанков
    num_chunks = length // chunk_size

    # Разбиваем аудио на чанки
    chunks = audios[:, :num_chunks * chunk_size].reshape(batch_size, num_chunks, chunk_size)

    # Разбиваем маски на чанки
    # [batch_ids, numchunks, chunk_size]
    chunks_mask = audios_mask[:, :num_chunks * chunk_size].reshape(batch_size, num_chunks, chunk_size)

    return chunks, chunks_mask

def audio_processor(chunk, mask):
    audio_inputs = {}
    audio_inputs['input_values'] = chunk  # Возвращаем тензор, а не numpy
    audio_inputs['attention_mask'] = mask  # Возвращаем тензор, а не numpy
    audio_inputs['decoder_input_ids'] = torch.zeros((chunk.shape[0], 1), dtype=torch.long)  # Возвращаем тензор

    return audio_inputs