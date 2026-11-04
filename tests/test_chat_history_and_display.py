import unittest
import uuid
from backend.conversations import (
    save_conversation,
    get_conversation,
    list_conversations,
    rename_conversation,
    delete_conversation,
    append_message_to_conversation
)


class TestChatHistoryAndDisplay(unittest.TestCase):

    def setUp(self):
        self.test_conv_id = f"test_conv_{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        delete_conversation(self.test_conv_id)

    def test_single_conversation_creation_and_append(self):
        """Rule 1 & 2: Stable unique ID, multi-turn updates the same conversation."""
        # Turn 1: Asthma
        turn1 = {
            "id": f"msg_{uuid.uuid4().hex[:6]}",
            "question": "What is asthma?",
            "answer": "Asthma is a chronic inflammatory disorder of the airways.",
            "confidence": "High",
            "safetyLevel": "Normal",
            "sources": [{"title": "Asthma Overview", "sources": "NHS"}]
        }
        res1 = append_message_to_conversation(self.test_conv_id, turn1)
        self.assertIsNotNone(res1)
        self.assertEqual(res1["id"], self.test_conv_id)
        self.assertEqual(len(res1["messages"]), 1)
        self.assertEqual(res1["messages"][0]["question"], "What is asthma?")

        # Turn 2: Follow-up in SAME conversation
        turn2 = {
            "id": f"msg_{uuid.uuid4().hex[:6]}",
            "question": "What are its triggers?",
            "answer": "Common triggers include pollen, cold air, smoke, and exercise.",
            "confidence": "High",
            "safetyLevel": "Normal",
            "sources": [{"title": "Asthma Triggers", "sources": "Mayo Clinic"}]
        }
        res2 = append_message_to_conversation(self.test_conv_id, turn2)
        self.assertEqual(len(res2["messages"]), 2)
        self.assertEqual(res2["id"], self.test_conv_id)

        # Retrieve and verify full conversation restored
        conv = get_conversation(self.test_conv_id)
        self.assertIsNotNone(conv)
        self.assertEqual(len(conv["messages"]), 2)
        self.assertEqual(conv["messages"][0]["question"], "What is asthma?")
        self.assertEqual(conv["messages"][1]["question"], "What are its triggers?")

    def test_no_duplicate_message_writes(self):
        """Rule 5 & 6: Do not save same message or assistant response more than once."""
        turn = {
            "id": "fixed_msg_id_123",
            "question": "Metformin side effects",
            "answer": "Nausea, diarrhea, stomach upset.",
            "confidence": "High"
        }
        append_message_to_conversation(self.test_conv_id, turn)
        # Attempt to append identical turn again
        append_message_to_conversation(self.test_conv_id, turn)

        conv = get_conversation(self.test_conv_id)
        self.assertEqual(len(conv["messages"]), 1)

    def test_save_conversation_does_not_wipe_messages(self):
        """Rule 4: Do not overwrite non-empty conversation with empty messages on sync."""
        turn = {
            "id": "msg_001",
            "question": "Asthma overview",
            "answer": "Airway inflammation details."
        }
        append_message_to_conversation(self.test_conv_id, turn)

        # Incoming empty sync object
        empty_sync = {
            "id": self.test_conv_id,
            "title": "Asthma overview",
            "messages": []
        }
        saved = save_conversation(empty_sync)
        # Must preserve the existing turn
        self.assertEqual(len(saved["messages"]), 1)
        self.assertEqual(saved["messages"][0]["question"], "Asthma overview")

    def test_rename_and_delete_isolation(self):
        """Rule 9: Renaming and deleting affect only the selected conversation."""
        other_conv_id = f"test_other_{uuid.uuid4().hex[:8]}"
        try:
            append_message_to_conversation(self.test_conv_id, {"question": "Q1", "answer": "A1"})
            append_message_to_conversation(other_conv_id, {"question": "Q2", "answer": "A2"})

            # Rename test_conv_id
            renamed = rename_conversation(self.test_conv_id, "Renamed Title")
            self.assertEqual(renamed["title"], "Renamed Title")

            # Check other conversation unchanged
            other = get_conversation(other_conv_id)
            self.assertNotEqual(other["title"], "Renamed Title")

            # Delete test_conv_id
            success = delete_conversation(self.test_conv_id)
            self.assertTrue(success)
            self.assertIsNone(get_conversation(self.test_conv_id))

            # Other conversation must still exist
            self.assertIsNotNone(get_conversation(other_conv_id))
        finally:
            delete_conversation(other_conv_id)


if __name__ == "__main__":
    unittest.main()
