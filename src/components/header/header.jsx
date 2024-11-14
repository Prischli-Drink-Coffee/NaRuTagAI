import React, { useRef } from "react";
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

const Header = () => {
  const btnRef = useRef();
  const { width, height } = useWindowDimensions();

  return (
    <Flex
      as="header"
      position="absolute"
      width={ width }
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
          <Image src={tr_logo} boxSize="300px" alt="Logo" position="absolute" top="-110px" left="120px" />
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
          display="flex"
          flexDirection="row"
          justifyContent="center"
          alignItems="center"
          padding="10px 20px"
          gap="10px"
          width="87px"
          height="42px"
          bg="#FFFFFF"
          borderRadius="10px"
          color="#1D1D1D"
          fontFamily="Montserrat"
          fontWeight="500"
          fontSize="18px"
          lineHeight="22px"
          _hover={{ bg: "#f0f0f0" }}
        >
          Вход
        </Button>

        {/* Кнопка "Регистрация" */}
        <Button
          display="flex"
          flexDirection="row"
          justifyContent="center"
          alignItems="center"
          padding="10px 20px"
          gap="10px"
          width="167px"
          height="42px"
          bg="#4B8BFC"
          borderRadius="10px"
          color="#FFFFFF"
          fontFamily="Montserrat"
          fontWeight="700"
          fontSize="18px"
          lineHeight="22px"
          _hover={{ bg: "#357ae8" }}
        >
          Регистрация
        </Button>
      </Flex>
    </Flex>
  );
};

export default Header;
