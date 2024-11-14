import { Text, VStack } from "@chakra-ui/react";
import useWindowDimensions from "../hooks/window_dimensions";
import UserService from "../API/services/user_service";
import { useState } from "react";
import { useEffect } from "react";

const MainPage = () => {
  const { width, height } = useWindowDimensions();
  const [admin, setAdmin] = useState(false);
  const getRole = async () => {
    const response = await UserService.me();
    if (response.data.userRole == "ADMIN") setAdmin(true);
  };
  useEffect(() => {
    getRole();
  }, []);
  return (
    <VStack
      padding={25}
      alignItems="flex-start"
      spacing="20px"
      flexGrow={1}
      width={"100%"}
    >
    <VStack
      padding={25}
      alignItems="flex-start"
      spacing="40px"
      flexGrow={1}
      width="100%"
    >
      <Text
        color="#000"
        fontSize='22px !important'
        fontWeight={700}
        lineHeight="normal"
        fontStyle="normal"
      >
        Главная страница
      </Text>
      <Text
        fontSize={14}
        fontWeight={400}
        marginBottom="20px"
        width={width > 720 ? "50%" : "100%"}
      >
        {admin
          ? `На главной странице вы можете видеть недавние действия пользователей,
        которые работают с вашим складом. Это поможет вам быть в курсе всех
        изменений, которые происходят на складе, а также контролировать работу
        своих сотрудников.`
          : `Добро пожалать в складскую систему! 
          Здесь вы можете управлять своими складскими запасами, отслеживать поступления и расходы товаров, а также просматривать статистику и отчеты по вашему бизнесу. Для того, чтобы перейти к другим разделам системы, вы можете использовать меню в верхней части страницы. Там вы найдете ссылки на различные разделы`}
      </Text>
    </VStack>
    </VStack>
  );
};
export default MainPage;
