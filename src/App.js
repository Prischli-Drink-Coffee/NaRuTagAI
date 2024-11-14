import { createHashRouter, RouterProvider } from "react-router-dom";

import Layout from "./Layout";
import SignInPage from "./pages/sign_in_page";
import SignUpPage from "./pages/sign_up_page";
import MainPage from "./pages/main_page";
import NotFoundPage from "./pages/notfound_page";
import PrivateRoutes from "./common/private_Routes";

const router = createHashRouter([
  {
    element: <Layout />,
    children: [
      {
        path: "/sign_up",
        element: <SignUpPage />,
      },
      {
        path: "/sign_in",
        element: <SignInPage />,
      },
      {
        element: <PrivateRoutes userGroup="AUTH" />,
        children: [
          {
            path: "/",
            element: <MainPage />,
            errorElement: <NotFoundPage />,
          }
        ],
      },
      {
        element: <PrivateRoutes userGroup="ADMIN" />,
        children: [],
      },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;
