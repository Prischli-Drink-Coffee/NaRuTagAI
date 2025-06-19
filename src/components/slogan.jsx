import React from 'react';
import { Box, Flex, Text, VStack } from "@chakra-ui/react";

const TagCloud = () => {
    return (
        <Box
            position="absolute"
            width="700px"
            height="600px"
            left="100px"
            p="4"
            bg="transpose"
        >

        <VStack spacing="55" mt="100px">
            {/* ��������� ���������� */}
            <Text
                fontFamily="Montserrat"
                fontWeight="700"
                fontSize="32px"
                lineHeight="39px"
                color="#023BA3"
            >
               NaRuTagAI - it's about video tags
            </Text>

            {/* �������������� ����� */}
            <Text
                fontFamily="Montserrat"
                fontWeight="500"
                fontSize="24px"
                lineHeight="29px"
                color="#1D1D1D"
                textAlign="center"
                maxWidth="400px"
            >
               Система NaRuTagAI автоматически генерирует теги для видео, используя передовые технологии искусственного интеллекта. Она анализирует содержание видео, извлекает ключевые моменты и создает релевантные теги, что значительно упрощает процесс поиска и организации видео-контента.
            </Text>
            
        </VStack>
        </Box>
    );
};

export default TagCloud;
