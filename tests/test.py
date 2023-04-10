import unittest
import functions


class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(functions.add(2, 4), 6)


if __name__ == '__main__':
    unittest.main()
