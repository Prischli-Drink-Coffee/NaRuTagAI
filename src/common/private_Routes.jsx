import { Navigate, Outlet } from "react-router-dom";
import { useCookies } from "react-cookie";

const PrivateRoutes = ({ userGroup }) => {
  const [cookie, setCookie] = useCookies();

  let isAllowed = false;

  if ((userGroup === "AUTH" && cookie.role) || userGroup === cookie.role) {
    isAllowed = true;
  }
  return isAllowed ? <Outlet /> : <Navigate to="/sign_up" />;
};

export default PrivateRoutes;
