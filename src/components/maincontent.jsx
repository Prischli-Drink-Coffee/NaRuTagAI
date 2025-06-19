import { Box, Flex, Text, Button, FormControl, Input, Spinner } from "@chakra-ui/react";
import { useBreakpointValue } from "@chakra-ui/react";
import { useState } from "react";
import useWindowDimensions from "../hooks/window_dimensions";

const ContentSection = ({ onFetch }) => {
  const { height } = useWindowDimensions();
  const [url, setUrl] = useState(""); // ��������� ��� URL
  const [isLoading, setIsLoading] = useState(false); // ��������� ��� ���������� ��������
  const buttonText = useBreakpointValue({
    base: "Ну и про что там?",
    sm: "Затегать?",
  });

  const handleFetch = () => {
    if (url.trim()) {
      setIsLoading(true); // ������������� ��������� �������� � true
      onFetch(url).finally(() => setIsLoading(false)); // ����� ���������� ��������� ��������� ��������� ��������
    }
  };

  return (
    <Box maxW="840px" width="100%" position="relative" bg="#ffffff" p={{ base: "20px", sm: "30px" }}>
      <Flex direction="column" align="flex-start" gap={height * 0.045}>
        <Text
          width={{ base: "100%", sm: "100%" }}
          fontFamily="Montserrat"
          fontWeight="700"
          fontSize={{ base: "24px", sm: "32px" }}
          lineHeight="46px"
          color="#023BA3"
        >
          Лень смотреть видео? Давайте глянем, про что оно
        </Text>
        <Text
          width={{ base: "100%", sm: "100%" }}
          fontFamily="Montserrat"
          fontWeight="500"
          fontSize={{ base: "16px", sm: "18px" }}
          lineHeight="22px"
          color="#1D1D1D"
        >
          Хватит гадать с тегами! Наша нейросеть проанализирует ваше видео, выделит ключевые моменты и создаст идеальные теги для максимального охвата. Один клик — и ваш контент увидят все.
        </Text>
        <Flex direction={{ base: "column", sm: "row" }} align="flex-start" gap="20px" w="100%">
          <FormControl id="URL" isRequired>
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)} // ��������� ������ URL
              width={{ base: "100%", sm: "100%" }}
              height="60px"
              border="4px solid #4B8BFC"
              borderRadius="16px"
              placeholder="Вставь в меня ссылочку, родненький"
              paddingLeft="20px"
              bg="#FFFFFF"
              _placeholder={{
                fontFamily: "Montserrat",
                fontWeight: "500",
                fontSize: "18px",
                lineHeight: "22px",
                color: "#1D1D1D",
              }}
            />
          </FormControl>
          <Button
            onClick={handleFetch} // ���������� �����
            width={{ base: "100%", sm: "240px" }}
            height="60px"
            background="#4B8BFC"
            borderRadius="16px"
            fontFamily="Montserrat"
            fontWeight="700"
            fontSize="18px"
            lineHeight="22px"
            color="#FFFFFF"
            _hover={{ background: "#376fcb" }}
            disabled={isLoading} // ��������� ������, ���� ���� ������� ��������
          >
            {isLoading ? (
              <Spinner size="sm" color="white" /> // ���������� ������� ��� ��������
            ) : (
              buttonText
            )}
          </Button>
        </Flex>
      </Flex>
    </Box>
  );
};

export default ContentSection;
