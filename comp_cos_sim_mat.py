import numpy as np
import argparse
import gzip

parser = argparse.ArgumentParser()
parser.add_argument('embed_file', help='词向量文件',
                    default='distance_module/zh.300.vec.gz')


def main():
    embeddings = []

    with gzip.open(args.embed_file) as fin:
        for idx, line in enumerate(fin):
            if idx == 0:
                continue
            embedding = [float(num) for num in line.decode('utf-8').strip().split()[1:]]
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    print(embeddings.T.shape)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.asarray(embeddings / norm, "float32")
    product = np.dot(embeddings, embeddings.T)
    np.save(('cos_sim_counter_fitting.npy'), product)

if __name__ == "__main__":
    args = parser.parse_args()
    main()
