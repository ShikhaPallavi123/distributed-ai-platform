import os
import tempfile
import unittest

from api.workspaces import WorkspaceStore, AccessDenied, QuotaExceeded


class TestWorkspaceStore(unittest.TestCase):
    def test_isolation_quota_and_logout(self):
        with tempfile.TemporaryDirectory() as directory:
            store = WorkspaceStore(os.path.join(directory, "workspace.sqlite3"), daily_limit=1)
            store.register("alice", "long-password-1")
            store.register("bob", "long-password-2")
            a_token = store.login("alice", "long-password-1")["token"]
            b_token = store.login("bob", "long-password-2")["token"]
            alice, bob = store.authenticate(a_token), store.authenticate(b_token)
            store.save_document(alice, "shared", "Apples are red.")
            store.save_document(bob, "shared", "Apples are green.")
            self.assertEqual(store.ask(alice, "apples")["citations"][0]["quote"], "Apples are red.")
            self.assertEqual(store.ask(bob, "apples")["citations"][0]["quote"], "Apples are green.")
            with self.assertRaises(QuotaExceeded):
                store.ask(alice, "apples")
            store.delete_document(bob, "shared")
            self.assertEqual(len(store.documents(alice)), 1)
            with self.assertRaises(AccessDenied):
                store.login("alice", "incorrect-password")
            store.logout(a_token)
            with self.assertRaises(AccessDenied):
                store.authenticate(a_token)
            with open(store.path, "rb") as handle:
                raw = handle.read()
            self.assertNotIn(b"long-password-1", raw)
            self.assertNotIn(a_token.encode(), raw)
