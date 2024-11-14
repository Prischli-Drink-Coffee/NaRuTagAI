import React, { useRef, useState } from "react";
import logo from "./../../images/logo.svg";
import {
    Flex,
    Image,
    Button,
    Text,
    Box
} from "@chakra-ui/react";
import useWindowDimensions from "../../hooks/window_dimensions";
import tr_logo from "../../images/transparent_logo.svg";
import { useNavigate } from "react-router";

const Header = () => {
    const btnRef = useRef();
    const navigate = useNavigate();
    const { width } = useWindowDimensions();

    // Создаем состояние для активной кнопки
    const [activeButton, setActiveButton] = useState("login");

    // Функции для навигации и изменения состояния кнопки
    const to_auth = () => {
        navigate("/sign_in");
        setActiveButton("login");
    };

    const to_reg = () => {
        navigate("/sign_up");
        setActiveButton("register");
    };

    // Задаем стили для активной и неактивной кнопок
    const activeStyle = {
        bg: "#4B8BFC",
        color: "#FFFFFF",
        fontWeight: "700"
    };

    const inactiveStyle = {
        bg: "#FFFFFF",
        color: "#1D1D1D",
        fontWeight: "500",
        _hover: { bg: "#f0f0f0" }
    };

    return (
        <Flex
            as="header"
            position="absolute"
            width={width}
            height="100px"
            left="50%"
            transform="translateX(-50%)"
            top="0"
            bg="#FFFFFF"
            align="center"
            justify="space-between"
            p="0"
        >
            <Box>
                <Image src={tr_logo} boxSize="300px" alt="Logo" position="absolute" top="-110px" left="110px" />
            </Box>

            {/* Логотип и название NaRuTagAI */}
            <Flex
                position="absolute"
                left="243px"
                top="50%"
                transform="translateY(-50%)"
                align="center"
                gap="8px"
            >
                <Image src={logo} boxSize="28px" alt="Logo" />
                <Text
                    fontFamily="Montserrat"
                    fontWeight="700"
                    fontSize="18px"
                    lineHeight="22px"
                    color="#4B8BFC"
                >
                    NaRuTagAI
                </Text>
            </Flex>

            {/* Кнопки навигации "Вход" и "Регистрация" */}
            <Flex
                display="flex"
                flexDirection="row"
                justifyContent="flex-end"
                alignItems="center"
                gap="30px"
                position="absolute"
                width="284px"
                height="42px"
                left="1113px"
                top="29px"
            >
                {/* Кнопка "Вход" */}
                <Button
                    {...(activeButton === "login" ? activeStyle : inactiveStyle)}  // Задаем стили в зависимости от активной кнопки
                    onClick={to_auth}  // Обновляем состояние при клике
                >
                    Вход
                </Button>

                {/* Кнопка "Регистрация" */}
                <Button
                    {...(activeButton === "register" ? activeStyle : inactiveStyle)}  // Задаем стили для другой кнопки
                    onClick={to_reg}  // Обновляем состояние при клике
                >
                    Регистрация
                </Button>
            </Flex>
        </Flex>
    );
};

export default Header;
