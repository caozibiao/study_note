import towhee


if __name__ == '__main__':
    # 创建 image embeddings 并构建索引
    towhee.glob['/Users/caozibiao/Downloads/niaochao/']('/Users/caozibiao/Downloads/niaochao/*.png')\
        .image_decode['/Users/caozibiao/Downloads/niaochao/', 'img']()\
        .image_text_embedding.clip['img', 'vec'](model_name='clip_vit_b32', modality='image')\
        .tensor_normalize['vec', 'vec']()\
        .to_faiss[('/Users/caozibiao/Downloads/niaochao/', 'vec')](findex='./index.bin')

    # 通过指定文本进行内容检索
    results = (
        towhee.dc['text'](['puppy Corgi'])
              .image_text_embedding.clip['text', 'vec'](model_name='clip_vit_b32', modality='text')
              .tensor_normalize['vec', 'vec']()
              .faiss_search['vec', 'results'](findex='./index.bin', k=3)
              .select['text', 'results']()
    )

    print(results)