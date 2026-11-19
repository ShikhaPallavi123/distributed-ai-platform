"""Tests for generative AI: text generation, diffusion, embeddings."""
import unittest
import numpy as np
from generative import (
    TextGenerator, Tokenizer, TokenizerMode,
    SimpleDiffusion, ImageVAE, NoiseScheduler, ScheduleType,
    Word2Vec, EmbeddingModel, SemanticSearch,
)


class TestTokenizer(unittest.TestCase):
    def test_char_tokenizer(self):
        tok = Tokenizer(mode=TokenizerMode.CHAR)
        tok.build_vocab("hello world")
        ids = tok.encode("hello")
        text = tok.decode(ids)
        self.assertEqual(text, "hello")

    def test_word_tokenizer(self):
        tok = Tokenizer(mode=TokenizerMode.WORD)
        tok.build_vocab("the cat sat on the mat")
        ids = tok.encode("the cat")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)


class TestTextGenerator(unittest.TestCase):
    def test_text_gen_creation(self):
        tok = Tokenizer(mode=TokenizerMode.CHAR)
        tok.build_vocab("abcdefghijklmnopqrstuvwxyz ")
        gen = TextGenerator(tok, d_model=32, n_heads=4, n_layers=1)
        self.assertIsNotNone(gen)

    def test_generate(self):
        tok = Tokenizer(mode=TokenizerMode.CHAR)
        tok.build_vocab("hello world abc xyz")
        gen = TextGenerator(tok, d_model=32, n_heads=4, n_layers=1, seed=42)
        text = gen.generate("hel", max_tokens=5)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


class TestDiffusion(unittest.TestCase):
    def test_noise_scheduler(self):
        ns = NoiseScheduler(num_timesteps=100, schedule=ScheduleType.LINEAR)
        betas = ns.betas
        self.assertEqual(len(betas), 100)
        self.assertTrue(np.all(betas > 0))

    def test_simple_diffusion(self):
        diff = SimpleDiffusion(data_dim=64, hidden_dim=32, num_timesteps=50, seed=42)
        samples = diff.generate(shape=(64,), num_steps=10)
        self.assertEqual(samples.shape, (64,))

    def test_image_vae(self):
        vae = ImageVAE(input_dim=64, hidden_dim=32, latent_dim=8, seed=42)
        x = np.random.rand(64)  # single sample
        recon, mu, logvar = vae.reconstruct(x)
        self.assertEqual(recon.shape, (64,))
        samples = vae.generate(num_samples=3)
        self.assertEqual(samples.shape, (3, 64))


class TestEmbeddings(unittest.TestCase):
    def test_word2vec_creation(self):
        w2v = Word2Vec(embedding_dim=16, window_size=2, seed=42)
        corpus = "the cat sat on the mat the dog ran in the park"
        w2v.train(corpus, epochs=5)
        vec = w2v.get_embedding("cat")
        self.assertEqual(vec.shape, (16,))

    def test_word2vec_similarity(self):
        w2v = Word2Vec(embedding_dim=16, seed=42)
        corpus = " ".join(["king queen man woman"] * 10)
        w2v.train(corpus, epochs=10)
        sim = w2v.most_similar("king", top_k=2)
        self.assertIsInstance(sim, list)

    def test_embedding_model(self):
        model = EmbeddingModel(vocab_size=50, embedding_dim=16)
        self.assertIsNotNone(model)

    def test_semantic_search(self):
        ss = SemanticSearch()
        ss.add("The cat sat on the mat")
        ss.add("The dog ran in the park")
        results = ss.search("cat sitting", top_k=2)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
