import React from "react";
import { VStack, Flex, Divider, Image } from "@chakra-ui/react";
import RegistrationForm from "../components/regform";
import TagCloud from "../components/tagcloud";
import { useNavigate } from "react-router";
import UserService from "../API/services/user_service";
import useWindowDimensions from "../hooks/window_dimensions";

const SignUpPage = () => {
  const navigate = useNavigate();
  const { width, height } = useWindowDimensions();

  const validate = (values) => {
    const errors = {};

    if (!values.email) {
      errors.email = "Required";
    }
    if (!values.password) {
      errors.password = "Required";
    }
    return errors;
  };

  const signUp = async (values) => {
    try {
      await UserService.signUp(values);
      alert("Пользователь зарегистрирован");
      navigate("/sign_in");
    } catch (error) {
      console.error("Error signUp:", error);
      alert("Ошибка при регистрации");
    }
  };

    return (
        <VStack minH="100vh" align="center" justify="center" bg="#ffffff">
        <Flex
        position="relative"
        width={ width }
        height="auto"
        align="center"
        justify="center"
        gap="20px"
        >
        {/* Форма регистрации слева с передачей функции signUp */}
        <RegistrationForm onSubmit={signUp} />

        {/* Разделительная линия */}
        <Divider
            orientation="vertical"
            border="1px solid rgba(75, 187, 252, 0.6)"
            height="600px"
        />

        {/* TagCloud справа */}
        <TagCloud />
        </Flex>
    </VStack>
  );
};

export default SignUpPage;
