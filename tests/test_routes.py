from __future__ import annotations

import re
import unittest
import urllib.parse

from tests import support


class PublicPageStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app, _ = support.load_app_and_service()
        cls.client = app.test_client()

    def test_home_page(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_browse_index(self):
        self.assertEqual(self.client.get("/browse").status_code, 200)

    def test_browse_drama(self):
        self.assertEqual(self.client.get("/browse/Drama").status_code, 200)

    def test_browse_childrens_genre(self):
        self.assertEqual(self.client.get("/browse/Children's").status_code, 200)

    def test_about_page(self):
        self.assertEqual(self.client.get("/about").status_code, 200)

    def test_health_check(self):
        self.assertEqual(self.client.get("/health").status_code, 200)


class UserBasedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app, _ = support.load_app_and_service()
        cls.client = app.test_client()

    def test_known_viewer_returns_200(self):
        response = self.client.post("/recommend/user", data={"user_id": "10"})
        self.assertEqual(response.status_code, 200)

    def test_unknown_viewer_returns_400(self):
        response = self.client.post("/recommend/user", data={"user_id": "9999"})
        self.assertEqual(response.status_code, 400)

    def test_viewer_zero_returns_400(self):
        response = self.client.post("/recommend/user", data={"user_id": "0"})
        self.assertEqual(response.status_code, 400)

    def test_non_numeric_viewer_returns_400(self):
        response = self.client.post("/recommend/user", data={"user_id": "abc"})
        self.assertEqual(response.status_code, 400)

    def test_empty_viewer_returns_400(self):
        response = self.client.post("/recommend/user", data={"user_id": ""})
        self.assertEqual(response.status_code, 400)


class RandomViewerRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app, service = support.load_app_and_service()
        cls.client = app.test_client()
        cls.known_user_ids = set(service.known_user_ids())

    def test_random_viewer_returns_200(self):
        self.assertEqual(self.client.get("/recommend/random").status_code, 200)

    def test_two_random_picks_are_known_viewers(self):
        for call in range(2):
            with self.subTest(call=call):
                body = self.client.get("/recommend/random").get_data(as_text=True)
                match = re.search(r"Recommendations for viewer (\d+)", body)
                self.assertIsNotNone(match)
                self.assertIn(int(match.group(1)), self.known_user_ids)


class ItemBasedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app, _ = support.load_app_and_service()
        cls.client = app.test_client()

    def test_known_title_post_returns_200(self):
        response = self.client.post("/recommend/movie", data={"movie_title": "Toy Story (1995)"})
        self.assertEqual(response.status_code, 200)

    def test_unknown_title_post_returns_400(self):
        response = self.client.post("/recommend/movie", data={"movie_title": "Not A Real Film (2099)"})
        self.assertEqual(response.status_code, 400)

    def test_boot_das_1981_page_returns_200(self):
        path = "/movie/" + urllib.parse.quote("Boot, Das (1981)")
        self.assertEqual(self.client.get(path).status_code, 200)

    def test_schindlers_list_page_returns_200(self):
        path = "/movie/" + urllib.parse.quote("Schindler's List (1993)")
        self.assertEqual(self.client.get(path).status_code, 200)


class BrowseAndNotFoundErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app, _ = support.load_app_and_service()
        cls.client = app.test_client()

    def test_unknown_genre_returns_400(self):
        self.assertEqual(self.client.get("/browse/NotAGenre").status_code, 400)

    def test_unknown_route_returns_404(self):
        self.assertEqual(self.client.get("/does-not-exist").status_code, 404)


class ServerErrorPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app, _ = support.load_app_and_service()
        cls.app.config["PROPAGATE_EXCEPTIONS"] = False
        cls.client = cls.app.test_client()

    def test_unhandled_exception_renders_error_template_without_exception_details(self):
        original = self.app.view_functions["api.health"]
        self.app.view_functions["api.health"] = self._raise_runtime_error
        try:
            response = self.client.get("/health")
        finally:
            self.app.view_functions["api.health"] = original

        self.assertEqual(response.status_code, 500)
        body = response.get_data(as_text=True)
        self.assertIn("Something went wrong on our side.", body)
        self.assertNotIn("RuntimeError", body)
        self.assertNotIn("Traceback", body)

    @staticmethod
    def _raise_runtime_error():
        raise RuntimeError("deliberately raised for test verification only")


if __name__ == "__main__":
    unittest.main()
